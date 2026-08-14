from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from grounded_qa.needle_tokenizer import NeedleTokenizer
from grounded_qa.needleish import NeedleConfig, NeedleishModel
from grounded_qa.synth_data import EncodedJsonlDataset, collate_synth


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def masked_stats(logits: torch.Tensor, batch: dict[str, torch.Tensor | list[dict]]) -> dict[str, float]:
    targets = batch["target_ids"]
    valid = batch["target_valid"]
    reasoning = valid & batch["reasoning_mask"]
    answer = valid & batch["answer_mask"]
    flat_logits = logits.float().reshape(-1, logits.shape[-1])
    flat_targets = targets.reshape(-1)
    total_loss = F.cross_entropy(flat_logits, flat_targets, ignore_index=0, reduction="sum")

    def part(mask: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        count = int(mask.sum())
        if count == 0:
            return torch.zeros((), device=logits.device), 0, 0
        selected_logits = logits[mask].float()
        selected_targets = targets[mask]
        loss = F.cross_entropy(selected_logits, selected_targets, reduction="sum")
        correct = int(selected_logits.argmax(dim=-1).eq(selected_targets).sum())
        return loss, count, correct

    reason_loss, reason_count, reason_correct = part(reasoning)
    answer_loss, answer_count, answer_correct = part(answer)
    total_count = int(valid.sum())
    return {
        "loss_total_sum": float(total_loss.detach()),
        "loss_reasoning_sum": float(reason_loss.detach()),
        "loss_answer_sum": float(answer_loss.detach()),
        "target_count": total_count,
        "reasoning_count": reason_count,
        "answer_count": answer_count,
        "reasoning_correct": reason_correct,
        "answer_correct": answer_correct,
    }


def merge_stats(total: dict[str, float], current: dict[str, float]) -> None:
    for key, value in current.items():
        total[key] = total.get(key, 0.0) + value


@torch.no_grad()
def evaluate(model: NeedleishModel, loader: DataLoader, device: torch.device, max_batches: int | None = None) -> dict[str, float]:
    model.eval()
    total: dict[str, float] = {}
    wrong_total: dict[str, float] = {}
    exercise: dict[str, dict[str, float]] = {}
    overlap: dict[str, dict[str, float]] = {}
    for index, batch in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            break
        tensors = {key: value.to(device) for key, value in batch.items() if isinstance(value, torch.Tensor)}
        logits = model(tensors["source_ids"], tensors["source_valid"], tensors["decoder_input_ids"], tensors["target_valid"])
        stats = masked_stats(logits, tensors)
        merge_stats(total, stats)
        wrong_source = tensors["source_ids"].roll(1, dims=0)
        wrong_valid = tensors["source_valid"].roll(1, dims=0)
        wrong_logits = model(wrong_source, wrong_valid, tensors["decoder_input_ids"], tensors["target_valid"])
        merge_stats(wrong_total, masked_stats(wrong_logits, tensors))
        for row, row_loss in zip(batch["metadata"], logits.float().unbind(0)):
            key = str(row["exercise"])
            bucket = "high" if row["answer_source_overlap"] >= 0.75 else "medium" if row["answer_source_overlap"] >= 0.25 else "low"
            for group, name in ((exercise, key), (overlap, bucket)):
                entry = group.setdefault(name, {"loss": 0.0, "tokens": 0.0, "answer_loss": 0.0, "answer_tokens": 0.0, "answer_correct": 0.0})
                target = torch.tensor(row["target_ids"], device=device)
                valid = target.ne(0)
                answer_mask = torch.tensor(row["answer_mask"], dtype=torch.bool, device=device) & valid
                value = F.cross_entropy(row_loss[: len(target)], target, reduction="sum")
                entry["loss"] += float(value)
                entry["tokens"] += float(valid.sum())
                if answer_mask.any():
                    answer_logits = row_loss[: len(target)][answer_mask]
                    answer_targets = target[answer_mask]
                    entry["answer_loss"] += float(F.cross_entropy(answer_logits, answer_targets, reduction="sum"))
                    entry["answer_tokens"] += float(answer_mask.sum())
                    entry["answer_correct"] += float(answer_logits.argmax(dim=-1).eq(answer_targets).sum())
    result = {
        "loss_total": total["loss_total_sum"] / max(total["target_count"], 1),
        "loss_reasoning": total["loss_reasoning_sum"] / max(total["reasoning_count"], 1),
        "loss_answer": total["loss_answer_sum"] / max(total["answer_count"], 1),
        "reasoning_token_accuracy": total["reasoning_correct"] / max(total["reasoning_count"], 1),
        "answer_token_accuracy": total["answer_correct"] / max(total["answer_count"], 1),
        "context_dependency_gap": (wrong_total["loss_total_sum"] / max(wrong_total["target_count"], 1)) - (total["loss_total_sum"] / max(total["target_count"], 1)),
    }
    for group_name, group in (("exercise", exercise), ("overlap", overlap)):
        for name, values in group.items():
            result[f"{group_name}/{name}/loss"] = values["loss"] / max(values["tokens"], 1)
            result[f"{group_name}/{name}/answer_loss"] = values["answer_loss"] / max(values["answer_tokens"], 1)
            result[f"{group_name}/{name}/answer_token_accuracy"] = values["answer_correct"] / max(values["answer_tokens"], 1)
    return result


@torch.no_grad()
def generate_probe(model: NeedleishModel, tokenizer: NeedleTokenizer, rows: list[dict], device: torch.device, max_new_tokens: int = 96) -> list[dict]:
    model.eval()
    outputs = []
    for row in rows:
        source = torch.tensor([row["source_ids"]], dtype=torch.long, device=device)
        source_valid = torch.ones_like(source, dtype=torch.bool)
        decoder = torch.tensor([[2]], dtype=torch.long, device=device)
        for _ in range(max_new_tokens):
            target_valid = torch.ones_like(decoder, dtype=torch.bool)
            logits = model(source, source_valid, decoder, target_valid)
            token = logits[:, -1].argmax(dim=-1, keepdim=True)
            decoder = torch.cat((decoder, token), dim=1)
            if int(token.item()) == 1:
                break
        generated = decoder[0, 1:].tolist()
        source_text = tokenizer.decode(row["source_ids"])
        target_text = tokenizer.decode(row["target_ids"])
        query = row.get("query") or source_text.split("<CONTEXT>", 1)[0].replace("<QUERY>", "").strip()
        context = row.get("context") or source_text.split("<CONTEXT>", 1)[-1].strip()
        gold_reasoning = row.get("gold_reasoning") or target_text.split("<ANSWER>", 1)[0].replace("<REASONING>", "").strip()
        gold_answer = row.get("gold_answer") or target_text.split("<ANSWER>", 1)[-1].replace("</s>", "").strip()
        outputs.append({
            "exercise": row["exercise"],
            "query": query,
            "context": context,
            "gold_reasoning": gold_reasoning,
            "gold_answer": gold_answer,
            "gold_target": target_text,
            "generated": tokenizer.decode(generated),
            "generated_ids": generated,
            "generated_tokens": len(generated),
            "eos": bool(generated and generated[-1] == 1),
        })
    return outputs


def probe_summary(outputs: list[dict]) -> dict[str, float]:
    repeated = []
    unique = []
    for output in outputs:
        ids = [token for token in output["generated_ids"] if token != 1]
        trigrams = [tuple(ids[index : index + 3]) for index in range(max(0, len(ids) - 2))]
        repeated.append(1.0 - len(set(trigrams)) / max(len(trigrams), 1))
        unique.append(len(set(ids)) / max(len(ids), 1))
    return {
        "eos_rate": sum(bool(output["eos"]) for output in outputs) / max(len(outputs), 1),
        "average_generated_length": sum(output["generated_tokens"] for output in outputs) / max(len(outputs), 1),
        "repeated_3gram_rate": sum(repeated) / max(len(repeated), 1),
        "unique_token_ratio": sum(unique) / max(len(unique), 1),
    }


@torch.no_grad()
def save_probe(model: NeedleishModel, tokenizer: NeedleTokenizer, rows: list[dict], device: torch.device, output_dir: Path, tokens_seen: int) -> dict[str, float]:
    outputs = generate_probe(model, tokenizer, rows, device, max_new_tokens=64)
    summary = probe_summary(outputs)
    with (output_dir / "probes.jsonl").open("a") as handle:
        handle.write(json.dumps({"tokens_seen": tokens_seen, "summary": summary, "examples": outputs}, ensure_ascii=False) + "\n")
    print(json.dumps({"tokens_seen": tokens_seen, **{f"probe/{key}": value for key, value in summary.items()}}), flush=True)
    return summary


def learning_rate(tokens_seen: int, max_tokens: int, max_lr: float) -> float:
    warmup = max(1, int(max_tokens * 0.02))
    decay_start = int(max_tokens * 0.80)
    if tokens_seen < warmup:
        return max_lr * tokens_seen / warmup
    if tokens_seen < decay_start:
        return max_lr
    progress = min(1.0, (tokens_seen - decay_start) / max(max_tokens - decay_start, 1))
    return max_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def checkpoint(path: Path, model: NeedleishModel, optimizer: torch.optim.Optimizer, *, step: int, tokens_seen: int, source_tokens_seen: int, target_tokens_seen: int, rows_seen: int, cfg: NeedleConfig, args: argparse.Namespace) -> None:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip()
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "tokens_seen": tokens_seen,
        "source_tokens_seen": source_tokens_seen,
        "target_tokens_seen": target_tokens_seen,
        "rows_seen": rows_seen,
        "config": cfg.to_dict(),
        "args": vars(args),
        "git_commit": commit,
        "tokenizer": args.tokenizer,
    }, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="artifacts/synth_day1")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output-dir", default="runs/needleish26m-day1")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=250_000_000)
    parser.add_argument("--max-hours", type=float, default=10.0)
    parser.add_argument("--max-val-batches", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--probe-n", type=int, default=50)
    args = parser.parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    cfg = NeedleConfig()
    model = NeedleishModel(cfg).to(device)
    print(json.dumps({"architecture": cfg.to_dict(), "parameters": model.n_params(), "breakdown": model.parameter_breakdown()}, indent=2))
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    start_step = 0
    tokens_seen = 0
    rows_seen = 0
    source_tokens_seen = 0
    target_tokens_seen = 0
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_step = state["step"]
        tokens_seen = state["tokens_seen"]
        source_tokens_seen = state.get("source_tokens_seen", 0)
        target_tokens_seen = state.get("target_tokens_seen", 0)
        rows_seen = state["rows_seen"]

    data_dir = Path(args.data_dir)
    train = EncodedJsonlDataset(data_dir / "train.jsonl")
    validation = EncodedJsonlDataset(data_dir / "validation.jsonl")
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0, collate_fn=collate_synth)
    val_loader = DataLoader(validation, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0, collate_fn=collate_synth)
    probe_rows = [validation[index] for index in range(min(args.probe_n, len(validation)))]
    tokenizer = NeedleTokenizer(args.tokenizer)

    wandb_run = None
    if args.wandb:
        import wandb
        wandb_run = wandb.init(project="grounded-attn-foundation", name="needleish26m-s42-synth-day1", config={**cfg.to_dict(), **vars(args), "parameters": model.n_params()})

    autocast_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}[args.precision]
    scaler = torch.amp.GradScaler("cuda", enabled=args.precision == "fp16")
    iterator = iter(train_loader)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    milestones = [1_000_000, 5_000_000, 10_000_000, 25_000_000, 50_000_000, 100_000_000, 150_000_000, 200_000_000, 250_000_000]
    next_milestone = next((index for index, value in enumerate(milestones) if value > tokens_seen), len(milestones))
    if tokens_seen == 0 and probe_rows:
        save_probe(model, tokenizer, probe_rows, device, output_dir, 0)
    last_log = started
    last_checkpoint = started
    step = start_step
    seen_ids: set[str] = set()
    while tokens_seen < args.max_tokens and (time.monotonic() - started) < args.max_hours * 3600:
        optimizer.zero_grad(set_to_none=True)
        batch_tokens = 0
        for _ in range(args.grad_accum):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                batch = next(iterator)
            tensors = {key: value.to(device, non_blocking=True) for key, value in batch.items() if isinstance(value, torch.Tensor)}
            source_batch_tokens = int(tensors["source_valid"].sum())
            target_batch_tokens = int(tensors["target_valid"].sum())
            batch_tokens += source_batch_tokens + target_batch_tokens
            source_tokens_seen += source_batch_tokens
            target_tokens_seen += target_batch_tokens
            rows_seen += len(batch["metadata"])
            seen_ids.update(row["row_id"] for row in batch["metadata"] if row["row_id"])
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
                logits = model(tensors["source_ids"], tensors["source_valid"], tensors["decoder_input_ids"], tensors["target_valid"])
                loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), tensors["target_ids"].reshape(-1), ignore_index=0) / args.grad_accum
            scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        current_lr = learning_rate(tokens_seen, args.max_tokens, 6e-4)
        for group in optimizer.param_groups:
            group["lr"] = current_lr
        scaler.step(optimizer)
        scaler.update()
        step += 1
        tokens_seen += batch_tokens
        now = time.monotonic()
        if now - last_log >= 30:
            elapsed = max(now - started, 1.0)
            metrics = {
                "train/loss_total": float(loss.detach()) * args.grad_accum,
                "train/grad_norm": float(grad_norm),
                "tokens_seen": tokens_seen,
                "rows_seen": rows_seen,
                "unique_rows_seen": len(seen_ids),
                "system/tokens_per_sec": tokens_seen / elapsed,
                "system/examples_per_sec": rows_seen / elapsed,
                "system/step_time": elapsed / max(step - start_step, 1),
                "system/gpu_allocated_gb": torch.cuda.memory_allocated(device) / 1e9 if device.type == "cuda" else 0.0,
                "system/gpu_reserved_gb": torch.cuda.memory_reserved(device) / 1e9 if device.type == "cuda" else 0.0,
                "system/memory_fraction": torch.cuda.memory_reserved(device) / torch.cuda.get_device_properties(device).total_memory if device.type == "cuda" else 0.0,
            }
            print(json.dumps({"step": step, **metrics}), flush=True)
            if wandb_run:
                wandb_run.log(metrics, step=tokens_seen)
            last_log = now
        if now - last_checkpoint >= 1800 or step == start_step + 1:
            checkpoint(output_dir / "latest.pt", model, optimizer, step=step, tokens_seen=tokens_seen, source_tokens_seen=source_tokens_seen, target_tokens_seen=target_tokens_seen, rows_seen=rows_seen, cfg=cfg, args=args)
            last_checkpoint = now
        if step % 500 == 0:
            metrics = evaluate(model, val_loader, device, args.max_val_batches)
            print(json.dumps({"step": step, **{f"val/{key}": value for key, value in metrics.items()}}), flush=True)
            if wandb_run:
                wandb_run.log({f"val/{key}": value for key, value in metrics.items()}, step=tokens_seen)
        while next_milestone < len(milestones) and tokens_seen >= milestones[next_milestone]:
            if probe_rows:
                save_probe(model, tokenizer, probe_rows, device, output_dir, milestones[next_milestone])
            checkpoint(output_dir / f"milestone-{milestones[next_milestone]}.pt", model, optimizer, step=step, tokens_seen=tokens_seen, source_tokens_seen=source_tokens_seen, target_tokens_seen=target_tokens_seen, rows_seen=rows_seen, cfg=cfg, args=args)
            next_milestone += 1
    checkpoint(output_dir / "latest.pt", model, optimizer, step=step, tokens_seen=tokens_seen, source_tokens_seen=source_tokens_seen, target_tokens_seen=target_tokens_seen, rows_seen=rows_seen, cfg=cfg, args=args)
    final = evaluate(model, val_loader, device, None)
    if probe_rows:
        save_probe(model, tokenizer, probe_rows, device, output_dir, tokens_seen)
    (output_dir / "final_metrics.json").write_text(json.dumps({"step": step, "tokens_seen": tokens_seen, "source_tokens_seen": source_tokens_seen, "target_tokens_seen": target_tokens_seen, "rows_seen": rows_seen, "unique_rows_seen": len(seen_ids), "elapsed_seconds": time.monotonic() - started, **final}, indent=2))
    if wandb_run:
        wandb_run.finish()
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
