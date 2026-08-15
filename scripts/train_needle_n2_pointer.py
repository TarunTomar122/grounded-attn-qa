from __future__ import annotations

import argparse
import json
import random
import subprocess
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from grounded_qa.calibration import choose_threshold, sweep_thresholds
from grounded_qa.metrics import exact_match, token_f1
from grounded_qa.negatives import REFUSAL
from grounded_qa.needle_pointer import NeedleAnswerablePointerModel, NeedlePointerModel, evidence_start_loss, evidence_start_targets, negative_eos_loss, pointer_loss
from grounded_qa.needle_tokenizer import EOS_ID, NeedleTokenizer
from grounded_qa.needleish import NeedleConfig
from scripts.train_needle_n1 import load_split, lr_at, optimizers_for


def batch(data: dict[str, torch.Tensor], indices: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, ...]:
    source = data["source_ids"][indices].to(device=device, dtype=torch.long, non_blocking=True)
    target = data["target_ids"][indices].to(device=device, dtype=torch.long, non_blocking=True)
    source_lengths = data["source_lengths"][indices].to(device=device, dtype=torch.long, non_blocking=True)
    target_lengths = data["target_lengths"][indices].to(device=device, dtype=torch.long, non_blocking=True)
    context_start = data["context_start"][indices].to(device=device, dtype=torch.long, non_blocking=True)
    gold = data["gold_copy_positions"][indices].to(device=device, dtype=torch.long, non_blocking=True)
    source_positions = torch.arange(source.shape[1], device=device)[None]
    source_valid = source_positions < source_lengths[:, None]
    context_mask = (source_positions >= context_start[:, None]) & source_valid
    target_valid = torch.arange(target.shape[1], device=device)[None] < target_lengths[:, None]
    decoder = torch.zeros_like(target)
    decoder[:, 0] = EOS_ID
    decoder[:, 1:] = target[:, :-1]
    labels = data.get("answerable")
    answerable = torch.ones(len(indices), dtype=torch.bool, device=device) if labels is None else labels[indices].to(device, non_blocking=True)
    return source, source_valid, context_mask, decoder, target, target_valid, gold, answerable


def swap_contexts(
    source: torch.Tensor,
    source_valid: torch.Tensor,
    context_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Keep each question fixed and replace only its context with the next row's."""
    if len(source) < 2:
        raise ValueError("Wrong-context evaluation needs at least two rows")
    positions = torch.arange(source.shape[1], device=source.device)[None]
    starts = context_mask.long().argmax(dim=-1)
    ranks = context_mask.long().cumsum(dim=-1) - 1
    packed = torch.zeros_like(source)
    rows, columns = context_mask.nonzero(as_tuple=True)
    packed[rows, ranks[rows, columns]] = source[rows, columns]
    donor = packed.roll(1, 0)
    donor_lengths = context_mask.sum(dim=-1).roll(1, 0).clamp_max(source.shape[1] - starts)
    destination_rank = positions - starts[:, None]
    wrong_context = (destination_rank >= 0) & (destination_rank < donor_lengths[:, None])
    gathered = donor.gather(1, destination_rank.clamp(0, source.shape[1] - 1))
    wrong_source = torch.where(wrong_context, gathered, torch.where(positions < starts[:, None], source, 0))
    wrong_valid = positions < (starts + donor_lengths)[:, None]
    return wrong_source, wrong_valid, wrong_context


def load_start(model: NeedlePointerModel | NeedleAnswerablePointerModel, path: Path, device: torch.device) -> None:
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        state = load_file(str(path), device=str(device))
    else:
        state = torch.load(path, map_location=device, weights_only=False)["model"]
    model.load_backbone_state_dict(state)


def calibrate_answerability(probabilities: torch.Tensor, answerable: torch.Tensor, steps: int = 201) -> dict[str, float]:
    """Report both a balanced-F1 threshold and the highest-coverage grounded threshold."""
    thresholds = torch.linspace(0, 1, steps)
    predicted = probabilities[:, None] >= thresholds[None]
    labels = answerable[:, None]
    tp = (predicted & labels).sum(dim=0).float()
    tn = (~predicted & ~labels).sum(dim=0).float()
    fp = (predicted & ~labels).sum(dim=0).float()
    fn = (~predicted & labels).sum(dim=0).float()
    answer_f1 = 2 * tp / (2 * tp + fp + fn).clamp_min(1)
    refusal_f1 = 2 * tn / (2 * tn + fp + fn).clamp_min(1)
    index = int(((answer_f1 + refusal_f1) / 2).argmax())
    grounded = choose_threshold(
        sweep_thresholds(probabilities.tolist(), answerable.tolist(), thresholds=steps),
        max_false_answer_rate=0.02,
    )
    return {
        "threshold": float(thresholds[index]),
        "answerability_f1": float(answer_f1[index]),
        "refusal_precision": float(tn[index] / (tn[index] + fn[index]).clamp_min(1)),
        "refusal_recall": float(tn[index] / (tn[index] + fp[index]).clamp_min(1)),
        "refusal_f1": float(refusal_f1[index]),
        "false_refusal_rate": float(fn[index] / (tp[index] + fn[index]).clamp_min(1)),
        "hallucinated_answer_rate": float(fp[index] / (tn[index] + fp[index]).clamp_min(1)),
        "grounded_threshold": grounded.threshold,
        "grounded_answer_coverage": grounded.answer_coverage,
        "grounded_answer_precision": grounded.answer_precision,
        "grounded_false_answer_rate": grounded.false_answer_rate,
        "grounded_false_refusal_rate": grounded.false_refusal_rate,
        "grounded_refusal_recall": grounded.refusal_recall,
    }


def answerability_bce_term(
    logits: torch.Tensor | None,
    answerable: torch.Tensor,
    *,
    pos_weight: torch.Tensor | None,
    weight: float,
) -> torch.Tensor:
    """Return the optional deployed answerability loss term."""
    if logits is None or weight == 0.0:
        return answerable.new_zeros((), dtype=torch.float32)
    return weight * F.binary_cross_entropy_with_logits(
        logits.float(), answerable.float(), pos_weight=pos_weight
    )


def negative_eos_term(
    output,
    source_ids: torch.Tensor,
    answerable: torch.Tensor,
    *,
    weight: float,
    eos_id: int = EOS_ID,
) -> torch.Tensor:
    """Return the optional immediate-EOS loss for unanswerable rows only."""
    if weight == 0.0:
        return output.vocab_logits.float().sum() * 0
    return weight * negative_eos_loss(output, source_ids, answerable, eos_id=eos_id)


@torch.inference_mode()
def evaluate(model, data, device, args) -> dict[str, float]:
    model.eval()
    all_probabilities: list[torch.Tensor] = []
    all_answerable: list[torch.Tensor] = []
    totals = {key: 0.0 for key in ("rows", "tokens", "weighted", "copy_tokens", "sequence", "z", "pointer", "pointer_correct", "gold_pointer_probability", "p_gen", "wrong_sequence", "wrong_weighted", "answerability_bce", "evidence_start", "evidence_correct", "evidence_positive_rows", "evidence_positive_correct", "evidence_negative_rows", "evidence_negative_correct", "negative_eos", "negative_eos_correct", "negative_eos_rows", "tp", "tn", "fp", "fn", "positive_probability", "negative_probability", "wrong_probability", "wrong_rows")}
    for start in range(0, len(data["source_ids"]), args.batch_size):
        indices = torch.arange(start, min(start + args.batch_size, len(data["source_ids"])))
        source, source_valid, context_mask, decoder, target, valid, gold, answerable = batch(data, indices, device)
        output = model(source, source_valid, context_mask, decoder, valid)
        loss_valid = valid & answerable[:, None] if args.answerability else valid
        loss = pointer_loss(
            output,
            source,
            target,
            loss_valid,
            gold,
            z_weight=args.z_loss,
            pointer_weight=args.pointer_weight,
            first_token_weight=args.first_token_weight,
            eos_token_weight=args.eos_token_weight,
        )
        wrong = wrong_loss = None
        if len(source) > 1:
            wrong_source, wrong_valid, wrong_context = swap_contexts(source, source_valid, context_mask)
            wrong = model(wrong_source, wrong_valid, wrong_context, decoder, valid)
            wrong_loss = pointer_loss(wrong, wrong_source, target, loss_valid, gold, z_weight=0.0, pointer_weight=0.0, first_token_weight=args.first_token_weight, eos_token_weight=args.eos_token_weight)
        rows = len(indices)
        tokens = int(loss_valid.sum())
        copy_tokens = int((loss_valid & gold.ge(0)).sum())
        weighted = float((loss_valid * (1 + torch.arange(target.shape[1], device=device).eq(0)[None] * (args.first_token_weight - 1) + target.eq(EOS_ID) * (args.eos_token_weight - 1))).sum())
        totals["rows"] += rows
        totals["tokens"] += tokens
        totals["weighted"] += weighted
        totals["copy_tokens"] += copy_tokens
        totals["sequence"] += float(loss.sequence) * weighted
        totals["z"] += float(loss.z) * tokens
        totals["pointer"] += float(loss.pointer_position) * copy_tokens
        totals["pointer_correct"] += float(loss.pointer_accuracy) * copy_tokens
        totals["gold_pointer_probability"] += float(loss.mean_gold_pointer_probability) * copy_tokens
        totals["p_gen"] += float(loss.mean_p_gen) * tokens
        if wrong_loss is not None:
            totals["wrong_sequence"] += float(wrong_loss.sequence) * weighted
            totals["wrong_weighted"] += weighted
        if output.answerability_logits is not None:
            totals["evidence_start"] += float(evidence_start_loss(output, gold, answerable)) * rows
            evidence_target = evidence_start_targets(gold, answerable)
            evidence_correct = output.evidence_position_logits.argmax(dim=-1).eq(evidence_target)
            totals["evidence_correct"] += int(evidence_correct.sum())
            totals["evidence_positive_rows"] += int(answerable.sum())
            totals["evidence_positive_correct"] += int((evidence_correct & answerable).sum())
            totals["evidence_negative_rows"] += int((~answerable).sum())
            totals["evidence_negative_correct"] += int((evidence_correct & ~answerable).sum())
            probability = output.answerability_logits.sigmoid()
            all_probabilities.append(probability.cpu())
            all_answerable.append(answerable.cpu())
            predicted = probability >= 0.5
            totals["answerability_bce"] += float(F.binary_cross_entropy_with_logits(output.answerability_logits, answerable.float(), reduction="sum"))
            totals["tp"] += int((predicted & answerable).sum())
            totals["tn"] += int((~predicted & ~answerable).sum())
            totals["fp"] += int((predicted & ~answerable).sum())
            totals["fn"] += int((~predicted & answerable).sum())
            totals["positive_probability"] += float((probability * answerable).sum())
            totals["negative_probability"] += float((probability * ~answerable).sum())
            negative_rows = int((~answerable).sum())
            negative_eos = negative_eos_loss(output, source, answerable, eos_id=EOS_ID)
            eos_distribution = output.final_distribution(source)[:, 0]
            totals["negative_eos"] += float(negative_eos) * negative_rows
            totals["negative_eos_correct"] += int((eos_distribution[~answerable].argmax(dim=-1) == EOS_ID).sum())
            totals["negative_eos_rows"] += negative_rows
            if wrong is not None and wrong.answerability_logits is not None:
                totals["wrong_probability"] += float(wrong.answerability_logits.sigmoid().sum())
                totals["wrong_rows"] += rows
    model.train()
    sequence = totals["sequence"] / max(totals["weighted"], 1)
    z = totals["z"] / max(totals["tokens"], 1)
    wrong = totals["wrong_sequence"] / max(totals["wrong_weighted"], 1)
    pointer = totals["pointer"] / max(totals["copy_tokens"], 1)
    metrics = {
        "val/loss": sequence + args.z_loss * z + args.pointer_weight * pointer,
        "val/sequence_nll": sequence,
        "val/z": z,
        "val/pointer_position_nll": pointer,
        "val/pointer_position_accuracy": totals["pointer_correct"] / max(totals["copy_tokens"], 1),
        "val/mean_gold_pointer_probability": totals["gold_pointer_probability"] / max(totals["copy_tokens"], 1),
        "val/mean_p_gen": totals["p_gen"] / max(totals["tokens"], 1),
        "val/wrong_context_sequence_nll": wrong,
        "val/context_dependency_gap": wrong - sequence,
    }
    if args.answerability:
        positives = totals["tp"] + totals["fn"]
        negatives = totals["tn"] + totals["fp"]
        precision = totals["tp"] / max(totals["tp"] + totals["fp"], 1)
        recall = totals["tp"] / max(positives, 1)
        refusal_precision = totals["tn"] / max(totals["tn"] + totals["fn"], 1)
        refusal_recall = totals["tn"] / max(negatives, 1)
        metrics.update({
            "val/loss": metrics["val/loss"] + args.answerability_weight * totals["evidence_start"] / totals["rows"] + args.negative_eos_weight * totals["negative_eos"] / max(totals["negative_eos_rows"], 1),
            "val/answerability_bce": totals["answerability_bce"] / totals["rows"],
            "val/evidence_start_nll": totals["evidence_start"] / totals["rows"],
            "val/evidence_choice_accuracy": totals["evidence_correct"] / totals["rows"],
            "val/evidence_start_accuracy": totals["evidence_positive_correct"] / max(totals["evidence_positive_rows"], 1),
            "val/evidence_no_answer_accuracy": totals["evidence_negative_correct"] / max(totals["evidence_negative_rows"], 1),
            "val/answerability_accuracy": (totals["tp"] + totals["tn"]) / totals["rows"],
            "val/answerability_precision": precision,
            "val/answerability_recall": recall,
            "val/answerability_f1": 2 * precision * recall / max(precision + recall, 1e-8),
            "val/false_refusal_rate": totals["fn"] / max(positives, 1),
            "val/refusal_precision": refusal_precision,
            "val/refusal_recall": refusal_recall,
            "val/refusal_f1": 2 * refusal_precision * refusal_recall / max(refusal_precision + refusal_recall, 1e-8),
            "val/hallucinated_answer_rate": totals["fp"] / max(negatives, 1),
            "val/mean_answerable_probability": totals["positive_probability"] / max(positives, 1),
            "val/mean_unanswerable_probability": totals["negative_probability"] / max(negatives, 1),
            "val/wrong_context_answerable_probability": totals["wrong_probability"] / max(totals["wrong_rows"], 1),
            "val/negative_eos_loss": totals["negative_eos"] / max(totals["negative_eos_rows"], 1),
            "val/negative_eos_accuracy": totals["negative_eos_correct"] / max(totals["negative_eos_rows"], 1),
        })
        calibrated = calibrate_answerability(torch.cat(all_probabilities), torch.cat(all_answerable))
        metrics.update({f"val/calibrated_{key}": value for key, value in calibrated.items()})
    return metrics


@torch.inference_mode()
def probe(model: NeedlePointerModel, tokenizer: NeedleTokenizer, data, device, count: int = 8, threshold: float = 0.5) -> tuple[dict[str, float], list[dict]]:
    model.eval()
    labels = data.get("answerable")
    if labels is None:
        indices = torch.arange(min(count, len(data["source_ids"])))
    else:
        half = count // 2
        indices = torch.cat((labels.nonzero().flatten()[:half], (~labels).nonzero().flatten()[: count - half]))
    source, source_valid, context_mask, _, target, _, _, answerable = batch(data, indices, device)
    memory = model.encode(source, source_valid)
    decoder = torch.full((len(indices), 1), EOS_ID, dtype=torch.long, device=device)
    if isinstance(model, NeedleAnswerablePointerModel):
        selection = model.decode_pointer(decoder, memory, source, source_valid, context_mask, torch.ones_like(decoder, dtype=torch.bool))
        probabilities = model.classify_answerability(model.evidence_position_logits(selection.copy_position_logits)).sigmoid()
    else:
        probabilities = torch.ones(len(indices), device=device)
    answerability_enabled = isinstance(model, NeedleAnswerablePointerModel)
    answered = probabilities >= threshold
    finished = torch.zeros(len(indices), dtype=torch.bool, device=device)
    for _ in range(128):
        output = model.decode_pointer(decoder, memory, source, source_valid, context_mask, torch.ones_like(decoder, dtype=torch.bool))
        last = type(output)(
            output.vocab_logits[:, -1:],
            output.copy_position_probs[:, -1:],
            output.p_gen[:, -1:],
        )
        next_ids = last.final_distribution(source)[:, 0].argmax(dim=-1)
        next_ids = torch.where(finished, torch.full_like(next_ids, EOS_ID), next_ids)
        decoder = torch.cat((decoder, next_ids[:, None]), dim=1)
        finished |= next_ids.eq(EOS_ID)
        if finished.all():
            break
    rows = []
    for generated, gold, should_answer, label, probability in zip(decoder[:, 1:].cpu().tolist(), target.cpu().tolist(), answered.cpu().tolist(), answerable.cpu().tolist(), probabilities.cpu().tolist()):
        emitted_eos = EOS_ID in generated
        generated = generated[: generated.index(EOS_ID)] if emitted_eos else generated
        gold = gold[: gold.index(EOS_ID)] if EOS_ID in gold else [token for token in gold if token]
        prediction, answer = tokenizer.decode(generated).strip(), tokenizer.decode(gold).strip()
        refused = not should_answer or (answerability_enabled and not prediction)
        rows.append({"prediction": REFUSAL if refused else prediction, "raw_prediction": prediction, "gold": answer, "answerable": label, "p_answerable": probability, "refused": refused, "eos": emitted_eos, "tokens": len(generated)})
    model.train()
    answerable_rows = [row for row in rows if row["answerable"]]
    negative_rows = [row for row in rows if not row["answerable"]]
    false_refusals = sum(row["refused"] for row in answerable_rows)
    correct_refusals = sum(row["refused"] for row in negative_rows)
    refusal_precision = correct_refusals / max(correct_refusals + false_refusals, 1)
    refusal_recall = correct_refusals / max(len(negative_rows), 1)
    return {
        "probe/em": sum(exact_match(row["prediction"], row["gold"]) for row in answerable_rows) / max(len(answerable_rows), 1),
        "probe/token_f1": sum(token_f1(row["prediction"], row["gold"]) for row in answerable_rows) / max(len(answerable_rows), 1),
        "probe/eos_rate": sum(row["eos"] for row in rows) / len(rows),
        "probe/mean_tokens": sum(row["tokens"] for row in rows) / len(rows),
        "probe/refusal_precision": refusal_precision,
        "probe/refusal_recall": refusal_recall,
        "probe/refusal_f1": 2 * refusal_precision * refusal_recall / max(refusal_precision + refusal_recall, 1e-8),
        "probe/false_refusal_rate": false_refusals / max(len(answerable_rows), 1),
        "probe/hallucinated_answer_rate": sum(not row["refused"] for row in negative_rows) / max(len(negative_rows), 1),
        "probe/answerability_threshold": threshold,
    }, rows


def save(path: Path, model, optimizers, step: int, seen_tokens: int, args) -> None:
    eager = getattr(model, "_orig_mod", model)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip()
    torch.save({
        "model": eager.state_dict(),
        "optimizers": {name: optimizer.state_dict() for name, optimizer in optimizers.items()},
        "step": step,
        "seen_tokens": seen_tokens,
        "config": NeedleConfig.public_checkpoint().to_dict(),
        "args": vars(args),
        "git_commit": commit,
    }, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Matched N2 pointer-generator adaptation of public Needle.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--lr", type=float, default=3.0e-5)
    parser.add_argument("--muon-lr", type=float, default=0.02)
    parser.add_argument("--z-loss", type=float, default=1.0e-4)
    parser.add_argument("--pointer-weight", type=float, default=1.0)
    parser.add_argument("--first-token-weight", type=float, default=2.0)
    parser.add_argument("--eos-token-weight", type=float, default=2.0)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--run-name", default="needle26m-n2-pg")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--answerability-weight", type=float, default=0.0)
    parser.add_argument("--answerability-bce-weight", type=float, default=0.0)
    parser.add_argument("--negative-eos-weight", type=float, default=0.0)
    parser.add_argument("--answerability-pos-weight", type=float)
    parser.add_argument("--answerability", action="store_true")
    args = parser.parse_args()
    if args.answerability_weight and not args.answerability:
        parser.error("--answerability-weight requires --answerability")
    if args.answerability_bce_weight and not args.answerability:
        parser.error("--answerability-bce-weight requires --answerability")
    if args.negative_eos_weight and not args.answerability:
        parser.error("--negative-eos-weight requires --answerability")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    train, validation = load_split(args.data_dir, "train"), load_split(args.data_dir, "validation")
    if args.answerability and (train["answerable"] & train["gold_copy_positions"][:, 0].lt(0)).any():
        parser.error("--answerability requires every answerable training row to have a gold source start; filter paraphrastic rows first")
    answerability_pos_weight = None
    if args.answerability:
        positives = int(train["answerable"].sum())
        negatives = len(train["answerable"]) - positives
        if args.answerability_pos_weight is None:
            args.answerability_pos_weight = negatives / max(positives, 1)
        answerability_pos_weight = torch.tensor(args.answerability_pos_weight, device=device)
    model_class = NeedleAnswerablePointerModel if args.answerability else NeedlePointerModel
    eager = model_class(NeedleConfig.public_checkpoint()).to(device=device, dtype=torch.bfloat16)
    load_start(eager, args.checkpoint, device)
    if args.evaluate_only:
        metrics = evaluate(eager, validation, device, args)
        probe_metrics, probe_rows = probe(eager, NeedleTokenizer(args.tokenizer, append_markers=False), validation, device, threshold=metrics.get("val/calibrated_threshold", 0.5))
        print(json.dumps({**metrics, **probe_metrics, "examples": probe_rows[:2]}, ensure_ascii=False))
        return
    optimizers = optimizers_for(eager, args.lr, args.muon_lr)
    step = seen_tokens = 0
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        eager.load_state_dict(state["model"])
        for name, optimizer_state in state["optimizers"].items():
            optimizers[name].load_state_dict(optimizer_state)
        step, seen_tokens = state["step"], state.get("seen_tokens", 0)
    model = torch.compile(eager) if args.compile else eager
    epoch_steps = len(train["source_ids"]) // args.batch_size
    if epoch_steps == 0:
        parser.error("training split is smaller than one batch")
    total_steps = args.epochs * epoch_steps
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run = None
    if args.wandb:
        import wandb

        config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
        run = wandb.init(project="grounded-attn-qa", group="needle-rag", name=args.run_name, config={**config, "train_rows": len(train["source_ids"]), "validation_rows": len(validation["source_ids"]), "total_steps": total_steps})

    initial = evaluate(model, validation, device, args)
    probe_metrics, probe_rows = probe(eager, NeedleTokenizer(args.tokenizer, append_markers=False), validation, device, threshold=initial.get("val/calibrated_threshold", 0.5))
    print(json.dumps({"step": step, **initial, **probe_metrics, "examples": probe_rows[:2]}, ensure_ascii=False), flush=True)
    if run:
        run.log({**initial, **probe_metrics}, step=step)
    started = last_log = time.perf_counter()
    last_step = step
    last_tokens = seen_tokens
    while step < total_steps:
        epoch = step // epoch_steps
        start = (step % epoch_steps) * args.batch_size
        order = torch.randperm(len(train["source_ids"]), generator=torch.Generator().manual_seed(args.seed + epoch))
        indices = order[start : start + args.batch_size]
        source, source_valid, context_mask, decoder, target, valid, gold, answerable = batch(train, indices, device)
        adam_lr, muon_lr = lr_at(step, total_steps, args.lr), lr_at(step, total_steps, args.muon_lr)
        for name, rate in (("adam", adam_lr), ("muon", muon_lr)):
            optimizer = optimizers.get(name)
            if optimizer is None:
                continue
            for group in optimizer.param_groups:
                group["lr"] = rate
        for optimizer in optimizers.values():
            optimizer.zero_grad(set_to_none=True)
        output = model(source, source_valid, context_mask, decoder, valid)
        loss_valid = valid & answerable[:, None] if args.answerability else valid
        loss = pointer_loss(output, source, target, loss_valid, gold, z_weight=args.z_loss, pointer_weight=args.pointer_weight, first_token_weight=args.first_token_weight, eos_token_weight=args.eos_token_weight)
        evidence_loss = evidence_start_loss(output, gold, answerable) if output.evidence_position_logits is not None else loss.total.new_zeros(())
        evidence_target = evidence_start_targets(gold, answerable) if output.evidence_position_logits is not None else None
        answerability_bce = answerability_bce_term(
            output.answerability_logits,
            answerable,
            pos_weight=answerability_pos_weight,
            weight=args.answerability_bce_weight,
        )
        negative_eos_weighted = negative_eos_term(
            output,
            source,
            answerable,
            weight=args.negative_eos_weight,
            eos_id=EOS_ID,
        )
        negative_eos = (
            negative_eos_weighted / args.negative_eos_weight
            if args.negative_eos_weight
            else output.vocab_logits.float().sum() * 0
        )
        total_loss = loss.total + args.answerability_weight * evidence_loss + answerability_bce + negative_eos_weighted
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for optimizer in optimizers.values():
            optimizer.step()
        step += 1
        seen_tokens += int(source_valid.sum() + valid.sum())
        if step == 1 or step % 50 == 0:
            now = time.perf_counter()
            metrics = {
                "train/loss": float(total_loss.detach()),
                "train/sequence_nll": float(loss.sequence.detach()),
                "train/pointer_position_nll": float(loss.pointer_position.detach()),
                "train/pointer_position_accuracy": float(loss.pointer_accuracy.detach()),
                "train/mean_gold_pointer_probability": float(loss.mean_gold_pointer_probability.detach()),
                "train/mean_p_gen": float(loss.mean_p_gen.detach()),
                "train/evidence_start_nll": float(evidence_loss.detach()),
                "train/evidence_choice_accuracy": float(output.evidence_position_logits.argmax(dim=-1).eq(evidence_target).float().mean()) if evidence_target is not None else 0.0,
                "train/answerability_bce": float(F.binary_cross_entropy_with_logits(output.answerability_logits, answerable.float()).detach()) if output.answerability_logits is not None else 0.0,
                "train/answerability_bce_term": float(answerability_bce.detach()),
                "train/negative_eos_loss": float(negative_eos.detach()),
                "train/negative_eos_term": float(negative_eos_weighted.detach()),
                "train/answerability_accuracy": float((output.answerability_logits.ge(0) == answerable).float().mean()) if output.answerability_logits is not None else 0.0,
                "train/grad_norm": float(grad_norm),
                "train/adam_lr": adam_lr,
                "train/muon_lr": muon_lr,
                "system/actual_tokens_per_second": (seen_tokens - last_tokens) / max(now - last_log, 1e-6),
                "system/padded_tokens_per_second": ((step - last_step) * args.batch_size * 1536) / max(now - last_log, 1e-6),
                "system/run_seconds": now - started,
                "system/peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9,
            }
            print(json.dumps({"step": step, **metrics}), flush=True)
            if run:
                run.log(metrics, step=step)
            last_log, last_step, last_tokens = now, step, seen_tokens
        if step % args.eval_every == 0 or step == total_steps:
            metrics = evaluate(model, validation, device, args)
            probe_metrics, probe_rows = probe(eager, NeedleTokenizer(args.tokenizer, append_markers=False), validation, device, threshold=metrics.get("val/calibrated_threshold", 0.5))
            print(json.dumps({"step": step, **metrics, **probe_metrics, "examples": probe_rows[:2]}, ensure_ascii=False), flush=True)
            if run:
                run.log({**metrics, **probe_metrics}, step=step)
            last_log, last_step, last_tokens = time.perf_counter(), step, seen_tokens
        if step % args.checkpoint_every == 0 or step == total_steps:
            save(args.output_dir / f"step-{step:06d}.pt", model, optimizers, step, seen_tokens, args)
    if run:
        run.finish()


if __name__ == "__main__":
    main()
