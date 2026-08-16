from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import torch

from grounded_qa.needle_full_span_qa import NeedleFullSpanNullModel
from grounded_qa.needle_span_qa import best_spans, span_null_loss
from grounded_qa.needle_tokenizer import NeedleTokenizer
from grounded_qa.needleish import NeedleConfig, load_public_checkpoint
from scripts.evaluate_needle_span_null import score, summarize
from scripts.train_needle_span_null import answer_text, batch, load_metadata, load_rows, threshold_scores


def model_and_optimizer(args: argparse.Namespace, device: torch.device) -> tuple[NeedleFullSpanNullModel, torch.optim.Optimizer]:
    model = NeedleFullSpanNullModel(NeedleConfig.public_checkpoint()).to(device)
    load_public_checkpoint(model.backbone, args.checkpoint)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": args.backbone_lr},
            {
                "params": [
                    *model.pointer.parameters(),
                    model.null_start_key,
                    model.null_end_key,
                    model.null_start_bias,
                    model.null_end_bias,
                ],
                "lr": args.head_lr,
            },
        ],
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )
    return model, optimizer


def gate_indices(rows: dict[str, torch.Tensor], kind: str) -> torch.Tensor:
    answerable = torch.where(rows["answerable"])[0]
    unanswerable = torch.where(~rows["answerable"])[0]
    if kind == "answerable":
        return answerable[:64]
    if kind == "unanswerable":
        return unanswerable[:64]
    if kind == "mixed":
        return torch.cat((answerable[:32], unanswerable[:32]))
    raise ValueError(f"unknown gate: {kind}")


@torch.inference_mode()
def measure_gate(
    model: NeedleFullSpanNullModel,
    rows: dict[str, torch.Tensor],
    indices: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    model.eval()
    has_rows = no_rows = 0
    has_start = has_end = no_start = no_end = 0
    for offset in range(0, len(indices), batch_size):
        chunk = indices[offset : offset + batch_size]
        source, valid, context, gold_start, gold_end, answerable = batch(rows, chunk, device)
        output = model(source, valid, context)
        predicted_start = output.start_logits.argmax(dim=1)
        predicted_end = output.end_logits.argmax(dim=1)
        has = answerable.bool()
        no = ~has
        has_rows += int(has.sum())
        no_rows += int(no.sum())
        has_start += int((predicted_start[has] == gold_start[has]).sum())
        has_end += int((predicted_end[has] == gold_end[has]).sum())
        no_start += int((predicted_start[no] == 0).sum())
        no_end += int((predicted_end[no] == 0).sum())
    return {
        "has_answer/start_accuracy": has_start / max(has_rows, 1),
        "has_answer/end_accuracy": has_end / max(has_rows, 1),
        "no_answer/start_accuracy": no_start / max(no_rows, 1),
        "no_answer/end_accuracy": no_end / max(no_rows, 1),
        "rows": float(len(indices)),
    }


def run_gates(
    args: argparse.Namespace,
    train_rows: dict[str, torch.Tensor],
    device: torch.device,
    precision: torch.dtype | None,
    log,
) -> None:
    for gate_number, kind in enumerate(("answerable", "unanswerable", "mixed")):
        indices = gate_indices(train_rows, kind)
        if len(indices) < 32:
            raise RuntimeError(f"gate {kind} has too few examples: {len(indices)}")
        model, optimizer = model_and_optimizer(args, device)
        generator = torch.Generator().manual_seed(args.seed + gate_number)
        model.train()
        for _ in range(args.gate_steps):
            sampled = indices[torch.randint(len(indices), (args.batch_size,), generator=generator)]
            source, valid, context, gold_start, gold_end, _ = batch(train_rows, sampled, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=precision, enabled=precision is not None and device.type == "cuda"):
                output = model(source, valid, context)
                loss, _, _ = span_null_loss(output, gold_start, gold_end)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        metrics = measure_gate(model, train_rows, indices, device, args.batch_size)
        log({f"gate/{kind}/{key}": value for key, value in metrics.items()}, gate_number)
        relevant = (
            metrics["has_answer/start_accuracy"] > 0.90 and metrics["has_answer/end_accuracy"] > 0.90
            if kind == "answerable"
            else metrics["no_answer/start_accuracy"] > 0.90 and metrics["no_answer/end_accuracy"] > 0.90
            if kind == "unanswerable"
            else min(
                metrics["has_answer/start_accuracy"],
                metrics["has_answer/end_accuracy"],
                metrics["no_answer/start_accuracy"],
                metrics["no_answer/end_accuracy"],
            ) > 0.80
        )
        if not relevant:
            raise RuntimeError(f"mechanical gate failed: {kind} {metrics}")
        del model, optimizer
        if device.type == "cuda":
            torch.cuda.empty_cache()


@torch.inference_mode()
def evaluate(
    model: NeedleFullSpanNullModel,
    rows: dict[str, torch.Tensor],
    metadata: list[dict[str, Any]],
    tokenizer: NeedleTokenizer,
    device: torch.device,
    batch_size: int,
    precision: torch.dtype | None,
    *,
    max_rows: int | None = None,
) -> dict[str, float]:
    model.eval()
    limit = min(max_rows, len(metadata)) if max_rows is not None else len(metadata)
    losses: list[float] = []
    raw_predictions: list[str] = []
    margins: list[float] = []
    has_rows = no_rows = 0
    has_start = has_end = no_start = no_end = 0
    for offset in range(0, limit, batch_size):
        indices = torch.arange(offset, min(offset + batch_size, limit))
        source, valid, context, gold_start, gold_end, answerable = batch(rows, indices, device)
        with torch.autocast(device_type="cuda", dtype=precision, enabled=precision is not None and device.type == "cuda"):
            output = model(source, valid, context)
            loss, _, _ = span_null_loss(output, gold_start, gold_end)
        losses.append(float(loss))
        best_start, best_end, _, margin = best_spans(output)
        argmax_start = output.start_logits.argmax(dim=1)
        argmax_end = output.end_logits.argmax(dim=1)
        has = answerable.bool()
        no = ~has
        has_rows += int(has.sum())
        no_rows += int(no.sum())
        has_start += int((argmax_start[has] == gold_start[has]).sum())
        has_end += int((argmax_end[has] == gold_end[has]).sum())
        no_start += int((argmax_start[no] == 0).sum())
        no_end += int((argmax_end[no] == 0).sum())
        for local, start, end, row_margin in zip(range(len(indices)), best_start.tolist(), best_end.tolist(), margin.tolist()):
            raw_predictions.append(answer_text(tokenizer, source[local].cpu(), start, end))
            margins.append(float(row_margin))

    eval_metadata = metadata[:limit]
    best_threshold = threshold_scores(margins, raw_predictions, eval_metadata)
    threshold_summary = summarize(eval_metadata, raw_predictions, margins, best_threshold["threshold"])
    raw_summary = summarize(eval_metadata, raw_predictions, [0.0] * len(margins), float("inf"))
    has_margin = [margins[i] for i, row in enumerate(eval_metadata) if row["answerable"]]
    no_margin = [margins[i] for i, row in enumerate(eval_metadata) if not row["answerable"]]
    return {
        "loss": sum(losses) / max(len(losses), 1),
        "has_answer/start_accuracy": has_start / max(has_rows, 1),
        "has_answer/end_accuracy": has_end / max(has_rows, 1),
        "no_answer/start_accuracy": no_start / max(no_rows, 1),
        "no_answer/end_accuracy": no_end / max(no_rows, 1),
        "raw/has_answer_em": raw_summary.get("has_answer_em", 0.0),
        "raw/has_answer_f1": raw_summary.get("has_answer_f1", 0.0),
        "threshold": best_threshold["threshold"],
        "threshold/em": threshold_summary.get("all_em", 0.0),
        "threshold/f1": threshold_summary.get("all_f1", 0.0),
        "threshold/has_answer_em": threshold_summary.get("has_answer_em", 0.0),
        "threshold/has_answer_f1": threshold_summary.get("has_answer_f1", 0.0),
        "threshold/no_answer_accuracy": threshold_summary["no_answer_accuracy"],
        "threshold/false_refusal_rate": threshold_summary["false_refusal_rate"],
        "threshold/false_answer_rate": threshold_summary["false_answer_rate"],
        "margin/has_answer_mean": sum(has_margin) / max(len(has_margin), 1),
        "margin/no_answer_mean": sum(no_margin) / max(len(no_margin), 1),
        "rows": float(limit),
    }


def save_checkpoint(path: Path, model, optimizer, step: int, metrics: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step, "metrics": metrics}, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train full pretrained Needle encoder-decoder grounded span + NULL on SQuAD2.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--full-eval-every", type=int, default=2000)
    parser.add_argument("--quick-eval-rows", type=int, default=1024)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--gate-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="grounded-attn-qa")
    parser.add_argument("--wandb-run-name", default="needle-full-decoder-span-null-squad2")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--skip-gates", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    precision = torch.bfloat16 if args.precision == "bf16" and device.type == "cuda" else None
    tokenizer = NeedleTokenizer(args.tokenizer, append_markers=False)
    train_rows = load_rows(args.data_dir / "train.pt")
    validation_rows = load_rows(args.data_dir / "validation.pt")
    validation_metadata = load_metadata(args.data_dir / "validation.jsonl")
    wandb_run = None
    if args.wandb:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        )

    def log(metrics: dict[str, float], step: int) -> None:
        if wandb_run is not None:
            wandb_run.log(metrics, step=step)
        print(json.dumps({"step": step, **metrics}, sort_keys=True), flush=True)

    try:
        if not args.skip_gates:
            run_gates(args, train_rows, device, precision, log)

        model, optimizer = model_and_optimizer(args, device)
        start_step = 0
        if args.resume:
            state = torch.load(args.resume, map_location="cpu", weights_only=False)
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            start_step = int(state["step"])

        quick_zero = evaluate(model, validation_rows, validation_metadata, tokenizer, device, args.batch_size, precision, max_rows=args.quick_eval_rows)
        log({f"quick/{key}": value for key, value in quick_zero.items()}, start_step)
        best_f1 = -1.0
        order = torch.randperm(len(train_rows["source_ids"]), generator=torch.Generator().manual_seed(args.seed))
        cursor = 0
        for step in range(start_step + 1, args.max_steps + 1):
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

            if step % args.eval_every == 0 or step == 1:
                quick = evaluate(model, validation_rows, validation_metadata, tokenizer, device, args.batch_size, precision, max_rows=args.quick_eval_rows)
                log({"train/loss": float(loss), "train/start_loss": float(start_loss), "train/end_loss": float(end_loss), "train/grad_norm": float(grad_norm), **{f"quick/{key}": value for key, value in quick.items()}}, step)
            if step % args.full_eval_every == 0 or step == args.max_steps:
                full = evaluate(model, validation_rows, validation_metadata, tokenizer, device, args.batch_size, precision)
                log({f"val/{key}": value for key, value in full.items()}, step)
                if full["threshold/f1"] > best_f1:
                    best_f1 = full["threshold/f1"]
                    save_checkpoint(args.output_dir / "best.pt", model, optimizer, step, full)
            if step % args.checkpoint_every == 0:
                save_checkpoint(args.output_dir / f"step-{step:06d}.pt", model, optimizer, step, full if "full" in locals() else quick)
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
