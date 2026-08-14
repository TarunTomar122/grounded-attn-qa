from __future__ import annotations

import argparse
import json
import random
import subprocess
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from grounded_qa.metrics import exact_match, token_f1
from grounded_qa.needle_pointer import NeedleAnswerablePointerModel, NeedlePointerModel, pointer_loss
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
    wrong_source = source.clone()
    wrong_valid = source_valid.clone()
    wrong_context = context_mask.clone()
    donor = source.roll(1, 0)
    donor_context = context_mask.roll(1, 0)
    for row in range(len(source)):
        start = int(context_mask[row].nonzero()[0])
        tokens = donor[row][donor_context[row]][: source.shape[1] - start]
        wrong_source[row, start:] = 0
        wrong_source[row, start : start + len(tokens)] = tokens
        wrong_valid[row, start:] = False
        wrong_valid[row, start : start + len(tokens)] = True
        wrong_context[row, start:] = False
        wrong_context[row, start : start + len(tokens)] = True
    return wrong_source, wrong_valid, wrong_context


def load_start(model: NeedlePointerModel | NeedleAnswerablePointerModel, path: Path, device: torch.device) -> None:
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        state = load_file(str(path), device=str(device))
    else:
        state = torch.load(path, map_location=device, weights_only=False)["model"]
    model.load_backbone_state_dict(state)


@torch.inference_mode()
def evaluate(model, data, device, args) -> dict[str, float]:
    model.eval()
    totals = {key: 0.0 for key in ("rows", "tokens", "weighted", "copy_tokens", "sequence", "z", "pointer", "pointer_correct", "gold_pointer_probability", "p_gen", "wrong_sequence", "answerability_bce", "tp", "tn", "fp", "fn", "positive_probability", "negative_probability", "wrong_probability")}
    for start in range(0, len(data["source_ids"]), args.batch_size):
        indices = torch.arange(start, min(start + args.batch_size, len(data["source_ids"])))
        source, source_valid, context_mask, decoder, target, valid, gold, answerable = batch(data, indices, device)
        output = model(source, source_valid, context_mask, decoder, valid)
        loss_valid = valid & answerable[:, None] if args.answerability_weight else valid
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
        wrong_source, wrong_valid, wrong_context = swap_contexts(source, source_valid, context_mask)
        wrong = model(wrong_source, wrong_valid, wrong_context, decoder, valid)
        wrong_loss = pointer_loss(
            wrong,
            wrong_source,
            target,
            loss_valid,
            gold,
            z_weight=0.0,
            pointer_weight=0.0,
            first_token_weight=args.first_token_weight,
            eos_token_weight=args.eos_token_weight,
        )
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
        totals["wrong_sequence"] += float(wrong_loss.sequence) * weighted
        if output.answerability_logits is not None:
            probability = output.answerability_logits.sigmoid()
            predicted = probability >= 0.5
            totals["answerability_bce"] += float(F.binary_cross_entropy_with_logits(output.answerability_logits, answerable.float(), reduction="sum"))
            totals["tp"] += int((predicted & answerable).sum())
            totals["tn"] += int((~predicted & ~answerable).sum())
            totals["fp"] += int((predicted & ~answerable).sum())
            totals["fn"] += int((~predicted & answerable).sum())
            totals["positive_probability"] += float((probability * answerable).sum())
            totals["negative_probability"] += float((probability * ~answerable).sum())
            if wrong.answerability_logits is not None:
                totals["wrong_probability"] += float(wrong.answerability_logits.sigmoid().sum())
    model.train()
    sequence = totals["sequence"] / max(totals["weighted"], 1)
    z = totals["z"] / max(totals["tokens"], 1)
    wrong = totals["wrong_sequence"] / max(totals["weighted"], 1)
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
    if args.answerability_weight:
        positives = totals["tp"] + totals["fn"]
        negatives = totals["tn"] + totals["fp"]
        precision = totals["tp"] / max(totals["tp"] + totals["fp"], 1)
        recall = totals["tp"] / max(positives, 1)
        metrics.update({
            "val/loss": metrics["val/loss"] + args.answerability_weight * totals["answerability_bce"] / totals["rows"],
            "val/answerability_bce": totals["answerability_bce"] / totals["rows"],
            "val/answerability_accuracy": (totals["tp"] + totals["tn"]) / totals["rows"],
            "val/answerability_precision": precision,
            "val/answerability_recall": recall,
            "val/answerability_f1": 2 * precision * recall / max(precision + recall, 1e-8),
            "val/false_refusal_rate": totals["fn"] / max(positives, 1),
            "val/refusal_recall": totals["tn"] / max(negatives, 1),
            "val/hallucinated_answer_rate": totals["fp"] / max(negatives, 1),
            "val/mean_answerable_probability": totals["positive_probability"] / max(positives, 1),
            "val/mean_unanswerable_probability": totals["negative_probability"] / max(negatives, 1),
            "val/wrong_context_answerable_probability": totals["wrong_probability"] / totals["rows"],
        })
    return metrics


@torch.inference_mode()
def probe(model: NeedlePointerModel, tokenizer: NeedleTokenizer, data, device, count: int = 8) -> tuple[dict[str, float], list[dict]]:
    model.eval()
    indices = torch.arange(min(count, len(data["source_ids"])))
    source, source_valid, context_mask, _, target, _, _, _ = batch(data, indices, device)
    memory = model.encode(source, source_valid)
    decoder = torch.full((len(indices), 1), EOS_ID, dtype=torch.long, device=device)
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
    for generated, gold in zip(decoder[:, 1:].cpu().tolist(), target.cpu().tolist()):
        emitted_eos = EOS_ID in generated
        generated = generated[: generated.index(EOS_ID)] if emitted_eos else generated
        gold = gold[: gold.index(EOS_ID)] if EOS_ID in gold else [token for token in gold if token]
        prediction, answer = tokenizer.decode(generated).strip(), tokenizer.decode(gold).strip()
        rows.append({"prediction": prediction, "gold": answer, "eos": emitted_eos, "tokens": len(generated)})
    model.train()
    return {
        "probe/em": sum(exact_match(row["prediction"], row["gold"]) for row in rows) / len(rows),
        "probe/token_f1": sum(token_f1(row["prediction"], row["gold"]) for row in rows) / len(rows),
        "probe/eos_rate": sum(row["eos"] for row in rows) / len(rows),
        "probe/mean_tokens": sum(row["tokens"] for row in rows) / len(rows),
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
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    train, validation = load_split(args.data_dir, "train"), load_split(args.data_dir, "validation")
    model_class = NeedleAnswerablePointerModel if args.answerability_weight else NeedlePointerModel
    eager = model_class(NeedleConfig.public_checkpoint()).to(device=device, dtype=torch.bfloat16)
    load_start(eager, args.checkpoint, device)
    if args.evaluate_only:
        metrics = evaluate(eager, validation, device, args)
        probe_metrics, probe_rows = probe(eager, NeedleTokenizer(args.tokenizer, append_markers=False), validation, device)
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
    total_steps = epoch_steps
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run = None
    if args.wandb:
        import wandb

        config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
        run = wandb.init(project="grounded-attn-qa", group="needle-rag", name=args.run_name, config={**config, "train_rows": len(train["source_ids"]), "validation_rows": len(validation["source_ids"]), "total_steps": total_steps})

    initial = evaluate(model, validation, device, args)
    probe_metrics, probe_rows = probe(eager, NeedleTokenizer(args.tokenizer, append_markers=False), validation, device)
    print(json.dumps({"step": step, **initial, **probe_metrics, "examples": probe_rows[:2]}, ensure_ascii=False), flush=True)
    if run:
        run.log({**initial, **probe_metrics}, step=step)
    started = last_log = time.perf_counter()
    last_step = step
    last_tokens = seen_tokens
    order = torch.randperm(len(train["source_ids"]), generator=torch.Generator().manual_seed(args.seed))
    for start in range(step * args.batch_size, total_steps * args.batch_size, args.batch_size):
        indices = order[start : start + args.batch_size]
        source, source_valid, context_mask, decoder, target, valid, gold, answerable = batch(train, indices, device)
        adam_lr, muon_lr = lr_at(step, total_steps, args.lr), lr_at(step, total_steps, args.muon_lr)
        for group in optimizers["adam"].param_groups:
            group["lr"] = adam_lr
        for group in optimizers["muon"].param_groups:
            group["lr"] = muon_lr
        for optimizer in optimizers.values():
            optimizer.zero_grad(set_to_none=True)
        output = model(source, source_valid, context_mask, decoder, valid)
        loss_valid = valid & answerable[:, None] if args.answerability_weight else valid
        loss = pointer_loss(output, source, target, loss_valid, gold, z_weight=args.z_loss, pointer_weight=args.pointer_weight, first_token_weight=args.first_token_weight, eos_token_weight=args.eos_token_weight)
        answerability_loss = F.binary_cross_entropy_with_logits(output.answerability_logits, answerable.float()) if output.answerability_logits is not None else loss.total.new_zeros(())
        total_loss = loss.total + args.answerability_weight * answerability_loss
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
                "train/answerability_bce": float(answerability_loss.detach()),
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
            probe_metrics, probe_rows = probe(eager, NeedleTokenizer(args.tokenizer, append_markers=False), validation, device)
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
