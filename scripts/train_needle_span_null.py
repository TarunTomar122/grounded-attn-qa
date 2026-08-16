from __future__ import annotations

import argparse
import json
import math
import random
import re
import string
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from grounded_qa.needle_span_qa import NeedleSpanNullModel, best_spans, span_null_loss, threshold_predictions
from grounded_qa.needleish import NeedleConfig, load_public_checkpoint
from grounded_qa.needle_tokenizer import NeedleTokenizer


def load_rows(path: Path) -> dict[str, torch.Tensor]:
    return torch.load(path, map_location="cpu", weights_only=True)


def load_metadata(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def batch(rows: dict[str, torch.Tensor], indices: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, ...]:
    source = rows["source_ids"][indices].to(device=device, dtype=torch.long, non_blocking=True)
    lengths = rows["source_lengths"][indices].to(device=device, dtype=torch.long, non_blocking=True)
    context_start = rows["context_start"][indices].to(device=device, dtype=torch.long, non_blocking=True)
    positions = torch.arange(source.shape[1], device=device)[None]
    source_valid = positions < lengths[:, None]
    context_mask = (positions >= context_start[:, None]) & source_valid
    return (
        source,
        source_valid,
        context_mask,
        rows["gold_start"][indices].to(device=device, dtype=torch.long, non_blocking=True),
        rows["gold_end"][indices].to(device=device, dtype=torch.long, non_blocking=True),
        rows["answerable"][indices].to(device=device, non_blocking=True),
    )


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def token_f1(prediction: str, truth: str) -> float:
    predicted = normalize_answer(prediction).split()
    expected = normalize_answer(truth).split()
    if not predicted or not expected:
        return float(predicted == expected)
    overlap: dict[str, int] = {}
    for token in predicted:
        overlap[token] = overlap.get(token, 0) + 1
    common = 0
    for token in expected:
        if overlap.get(token, 0):
            overlap[token] -= 1
            common += 1
    if not common:
        return 0.0
    precision = common / len(predicted)
    recall = common / len(expected)
    return 2 * precision * recall / (precision + recall)


def answer_text(tokenizer: NeedleTokenizer, source: torch.Tensor, start: int, end: int) -> str:
    if start == 0 or end == 0 or end < start:
        return ""
    return tokenizer.decode(source[start - 1 : end].tolist()).strip()


def threshold_scores(
    margins: list[float],
    predictions: list[str],
    metadata: list[dict[str, Any]],
) -> dict[str, float]:
    candidates = sorted(set([min(margins), max(margins), *margins]))
    if len(candidates) > 101:
        step = max(1, len(candidates) // 100)
        candidates = candidates[::step]
    best = {"f1": -1.0, "threshold": 0.0, "em": 0.0}
    for threshold in candidates:
        scored: list[tuple[str, dict[str, Any]]] = []
        for margin, prediction, row in zip(margins, predictions, metadata):
            scored.append(("" if margin >= threshold else prediction, row))
        em = 0.0
        f1 = 0.0
        for prediction, row in scored:
            truths = row["answers"]["text"] or [""]
            em += max(float(normalize_answer(prediction) == normalize_answer(truth)) for truth in truths)
            f1 += max(token_f1(prediction, truth) for truth in truths)
        em /= max(len(scored), 1)
        f1 /= max(len(scored), 1)
        if f1 > best["f1"]:
            best = {"f1": f1, "threshold": threshold, "em": em}
    return best


@torch.inference_mode()
def evaluate(
    model: NeedleSpanNullModel,
    rows: dict[str, torch.Tensor],
    metadata: list[dict[str, Any]],
    tokenizer: NeedleTokenizer,
    device: torch.device,
    batch_size: int,
    precision: torch.dtype | None,
    *,
    threshold: float = 0.0,
) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    answerable_start = answerable_end = answerable_rows = 0
    null_start = null_end = unanswerable_rows = 0
    predictions: list[str] = []
    margins: list[float] = []
    for offset in range(0, len(rows["source_ids"]), batch_size):
        indices = torch.arange(offset, min(offset + batch_size, len(rows["source_ids"])))
        source, valid, context, gold_start, gold_end, answerable = batch(rows, indices, device)
        with torch.autocast(device_type="cuda", dtype=precision, enabled=precision is not None and device.type == "cuda"):
            output = model(source, valid, context)
            loss, _, _ = span_null_loss(output, gold_start, gold_end)
        losses.append(float(loss))
        predicted_start, predicted_end = threshold_predictions(output, threshold=threshold)
        _, _, _, margin = best_spans(output)
        argmax_start = output.start_logits.argmax(dim=1)
        argmax_end = output.end_logits.argmax(dim=1)
        answerable_mask = answerable.bool()
        negative_mask = ~answerable_mask
        answerable_rows += int(answerable_mask.sum())
        unanswerable_rows += int(negative_mask.sum())
        answerable_start += int((argmax_start[answerable_mask] == gold_start[answerable_mask]).sum())
        answerable_end += int((argmax_end[answerable_mask] == gold_end[answerable_mask]).sum())
        null_start += int((argmax_start[negative_mask] == 0).sum())
        null_end += int((argmax_end[negative_mask] == 0).sum())
        for row_index, start, end, row_margin in zip(indices.tolist(), predicted_start.tolist(), predicted_end.tolist(), margin.tolist()):
            predictions.append(answer_text(tokenizer, source[row_index - offset].cpu(), start, end))
            margins.append(float(row_margin))
    thresholded = threshold_scores(margins, predictions, metadata)
    return {
        "loss": sum(losses) / max(len(losses), 1),
        "has_answer/start_accuracy": answerable_start / max(answerable_rows, 1),
        "has_answer/end_accuracy": answerable_end / max(answerable_rows, 1),
        "no_answer/start_accuracy": null_start / max(unanswerable_rows, 1),
        "no_answer/end_accuracy": null_end / max(unanswerable_rows, 1),
        "threshold/em": thresholded["em"],
        "threshold/f1": thresholded["f1"],
        "threshold/value": thresholded["threshold"],
        "rows": len(rows["source_ids"]),
    }


def save_checkpoint(path: Path, model: NeedleSpanNullModel, optimizer: torch.optim.Optimizer, step: int, metrics: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step, "metrics": metrics}, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune public Needle jointly on SQuAD2 spans and NULL.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Public Needle .safetensors checkpoint.")
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--backbone-lr", type=float, default=2e-5)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="grounded-attn-qa")
    parser.add_argument("--wandb-run-name", default="needle-span-null-squad2")
    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    precision = torch.bfloat16 if args.precision == "bf16" and device.type == "cuda" else None

    tokenizer = NeedleTokenizer(args.tokenizer, append_markers=False)
    train_rows = load_rows(args.data_dir / "train.pt")
    validation_rows = load_rows(args.data_dir / "validation.pt")
    validation_metadata = load_metadata(args.data_dir / "validation.jsonl")
    model = NeedleSpanNullModel(NeedleConfig.public_checkpoint()).to(device)
    load_public_checkpoint(model.backbone, args.checkpoint)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": args.backbone_lr},
            {"params": [*model.start_head.parameters(), *model.end_head.parameters(), model.null_start_key, model.null_end_key, model.null_start_bias, model.null_end_bias], "lr": args.head_lr},
        ],
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )

    wandb_run = None
    if args.wandb:
        import wandb

        wandb_run = wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))

    def log(metrics: dict[str, float], step: int) -> None:
        if wandb_run is not None:
            wandb_run.log(metrics, step=step)
        print(json.dumps({"step": step, **metrics}, sort_keys=True), flush=True)

    step_zero = evaluate(model, validation_rows, validation_metadata, tokenizer, device, args.batch_size, precision)
    log({f"step0/{key}": value for key, value in step_zero.items()}, 0)
    model.train()
    order = torch.randperm(len(train_rows["source_ids"]), generator=torch.Generator().manual_seed(args.seed))
    cursor = 0
    best_f1 = -1.0
    for step in range(1, args.max_steps + 1):
        if cursor + args.batch_size > len(order):
            order = torch.randperm(len(train_rows["source_ids"]), generator=torch.Generator().manual_seed(args.seed + step))
            cursor = 0
        indices = order[cursor : cursor + args.batch_size]
        cursor += args.batch_size
        source, valid, context, gold_start, gold_end, _ = batch(train_rows, indices, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=precision, enabled=precision is not None and device.type == "cuda"):
            output = model(source, valid, context)
            loss, start_loss, end_loss = span_null_loss(output, gold_start, gold_end)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % args.eval_every == 0:
            train_metrics = {
                "train/loss": float(loss),
                "train/start_loss": float(start_loss),
                "train/end_loss": float(end_loss),
                "train/grad_norm": float(grad_norm),
            }
            log(train_metrics, step)
            validation_metrics = evaluate(model, validation_rows, validation_metadata, tokenizer, device, args.batch_size, precision)
            log({f"val/{key}": value for key, value in validation_metrics.items()}, step)
            if validation_metrics["threshold/f1"] > best_f1:
                best_f1 = validation_metrics["threshold/f1"]
                save_checkpoint(args.output_dir / "best.pt", model, optimizer, step, validation_metrics)
        if step % args.checkpoint_every == 0:
            save_checkpoint(args.output_dir / f"step-{step:06d}.pt", model, optimizer, step, validation_metrics if "validation_metrics" in locals() else {})
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
