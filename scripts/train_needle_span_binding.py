"""Teach N2-PG to separate matched question/span bindings while replaying QA."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from grounded_qa.needle_pointer import NeedlePointerModel, pointer_loss
from grounded_qa.needleish import NeedleConfig
from scripts.train_needle_n1 import load_split
from scripts.train_needle_n2_pointer import batch as reader_batch
from scripts.train_needle_n2_pointer import load_start
from scripts.train_needle_joint_verifier import evaluate_reader


def matched_pairs(data: dict[str, torch.Tensor]) -> torch.Tensor:
    """Pair an answerable row with an unanswerable question over identical context tokens."""
    groups: dict[bytes, list[int]] = defaultdict(list)
    for index, (source, length, context_start) in enumerate(zip(data["source_ids"], data["source_lengths"], data["context_start"])):
        groups[source[int(context_start) : int(length)].numpy().tobytes()].append(index)
    pairs = []
    for indices in groups.values():
        positive = next((index for index in indices if bool(data["answerable"][index]) and int(data["gold_copy_positions"][index, 0]) >= 0), None)
        negative = next((index for index in indices if not bool(data["answerable"][index])), None)
        if positive is not None and negative is not None:
            offset = int(data["gold_copy_positions"][positive, 0] - data["context_start"][positive])
            pairs.append((positive, negative, offset))
    return torch.tensor(pairs, dtype=torch.long)


def contrastive_loss(positive: torch.Tensor, negative: torch.Tensor, margin: float = 0.2, temperature: float = 1.0) -> torch.Tensor:
    """Rank the matched span above the same-context hard negative."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return F.softplus((negative - positive + margin) / temperature).mean()


def _mean(memory: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (memory * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)


def span_scores(model: NeedlePointerModel, data: dict[str, torch.Tensor], pairs: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    positive_index, negative_index, offset = (pairs[:, column] for column in range(3))
    positive_source = data["source_ids"][positive_index].to(device=device, dtype=torch.long)
    negative_source = data["source_ids"][negative_index].to(device=device, dtype=torch.long)
    positive_length = data["source_lengths"][positive_index].to(device=device, dtype=torch.long)
    negative_length = data["source_lengths"][negative_index].to(device=device, dtype=torch.long)
    positive_context = data["context_start"][positive_index].to(device=device, dtype=torch.long)
    negative_context = data["context_start"][negative_index].to(device=device, dtype=torch.long)
    positions = torch.arange(positive_source.shape[1], device=device)[None]
    positive_valid, negative_valid = positions < positive_length[:, None], positions < negative_length[:, None]
    positive_question, negative_question = positions < positive_context[:, None], positions < negative_context[:, None]
    gold = data["gold_copy_positions"][positive_index].to(device=device, dtype=torch.long)
    gold_valid = gold.ge(0)
    positive_span = torch.zeros_like(positive_valid)
    negative_span = torch.zeros_like(negative_valid)
    rows = torch.arange(len(pairs), device=device)[:, None].expand_as(gold)
    positive_span[rows[gold_valid], gold[gold_valid]] = True
    remapped = negative_context[:, None] + (gold - positive_context[:, None])
    negative_span[rows[gold_valid], remapped[gold_valid]] = True
    positive_memory, negative_memory = model.encode(positive_source, positive_valid), model.encode(negative_source, negative_valid)
    positive = F.cosine_similarity(_mean(positive_memory, positive_question), _mean(positive_memory, positive_span), dim=-1)
    negative = F.cosine_similarity(_mean(negative_memory, negative_question), _mean(negative_memory, negative_span), dim=-1)
    return positive, negative


@torch.inference_mode()
def evaluate_binding(model: NeedlePointerModel, data: dict[str, torch.Tensor], pairs: torch.Tensor, device: torch.device, batch_size: int, margin: float = 0.2, temperature: float = 1.0) -> dict[str, float]:
    model.eval()
    positives, negatives = [], []
    for start in range(0, len(pairs), batch_size):
        positive, negative = span_scores(model, data, pairs[start : start + batch_size], device)
        positives.append(positive.float().cpu())
        negatives.append(negative.float().cpu())
    model.train()
    positive, negative = torch.cat(positives), torch.cat(negatives)
    return {"binding/pair_accuracy": float(positive.gt(negative).float().mean()), "binding/positive_score": float(positive.mean()), "binding/negative_score": float(negative.mean()), "binding/loss": float(contrastive_loss(positive, negative, margin, temperature))}


def main() -> None:
    parser = argparse.ArgumentParser(description="Joint N2 pointer replay plus span-binding contrastive training.")
    parser.add_argument("--reader-data-dir", type=Path, required=True)
    parser.add_argument("--matched-data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2_000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--reader-eval-rows", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--binding-ratio", type=float, default=0.5)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--run-name", default="needle26m-span-binding-2000")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    reader_train, reader_validation = load_split(args.reader_data_dir, "train"), load_split(args.reader_data_dir, "validation")
    matched_train, matched_validation = load_split(args.matched_data_dir, "train"), load_split(args.matched_data_dir, "validation")
    pairs_train, pairs_validation = matched_pairs(matched_train), matched_pairs(matched_validation)
    if not len(pairs_train) or not len(pairs_validation):
        raise ValueError("matched data must contain answerable/unanswerable context pairs")
    model = NeedlePointerModel(NeedleConfig.public_checkpoint()).to(device=device, dtype=torch.bfloat16)
    load_start(model, args.checkpoint, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    generator = torch.Generator().manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run = None
    if args.wandb:
        import wandb

        run = wandb.init(project="grounded-attn-qa", group="needle-rag", name=args.run_name, config={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()})
    initial = {**evaluate_reader(model, reader_validation, device, args.batch_size, args.reader_eval_rows), **evaluate_binding(model, matched_validation, pairs_validation, device, args.batch_size, args.margin, args.temperature)}
    baseline_pointer = initial["reader/pointer_position_accuracy"]
    print(json.dumps({"step": 0, **initial}), flush=True)
    if run:
        run.log(initial, step=0)
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        if torch.rand((), generator=generator).item() < args.binding_ratio:
            indices = torch.randint(len(pairs_train), (args.batch_size,), generator=generator)
            positive, negative = span_scores(model, matched_train, pairs_train[indices], device)
            loss = contrastive_loss(positive, negative, args.margin, args.temperature)
            train_metrics = {"train/task": "binding", "train/binding_loss": float(loss.detach()), "train/binding_pair_accuracy": float(positive.gt(negative).float().mean())}
        else:
            indices = torch.randint(len(reader_train["source_ids"]), (args.batch_size,), generator=generator)
            source, valid_source, context, decoder, target, valid_target, gold, _ = reader_batch(reader_train, indices, device)
            loss_info = pointer_loss(model(source, valid_source, context, decoder, valid_target), source, target, valid_target, gold)
            loss = loss_info.total
            train_metrics = {"train/task": "answer", "train/answer_loss": float(loss.detach()), "train/pointer_position_accuracy": float(loss_info.pointer_accuracy.detach())}
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 50 == 0:
            train_metrics.update({"train/grad_norm": float(grad_norm), "system/steps_per_second": step / (time.perf_counter() - started), "system/peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9})
            print(json.dumps({"step": step, **train_metrics}), flush=True)
            if run:
                run.log(train_metrics, step=step)
        if step % args.eval_every == 0 or step == args.steps:
            metrics = {**evaluate_reader(model, reader_validation, device, args.batch_size, args.reader_eval_rows), **evaluate_binding(model, matched_validation, pairs_validation, device, args.batch_size, args.margin, args.temperature)}
            metrics["reader/pointer_accuracy_delta"] = metrics["reader/pointer_position_accuracy"] - baseline_pointer
            print(json.dumps({"step": step, **metrics}), flush=True)
            if run:
                run.log(metrics, step=step)
            torch.save({"model": model.state_dict(), "step": step, "metrics": metrics, "args": vars(args)}, args.output_dir / f"step-{step:06d}.pt")
    if run:
        run.finish()


if __name__ == "__main__":
    main()
