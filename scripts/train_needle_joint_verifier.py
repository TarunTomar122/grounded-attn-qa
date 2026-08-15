"""Jointly teach the N2 reader to answer and verify, with answer replay as a safety gate."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from grounded_qa.needle_pointer import NeedlePointerModel, pointer_loss
from grounded_qa.needle_verifier import joint_verifier_logits
from grounded_qa.needleish import NeedleConfig
from scripts.train_needle_n1 import load_split
from scripts.train_needle_n2_pointer import batch as reader_batch
from scripts.train_needle_n2_pointer import load_start
from scripts.train_needle_nli_verifier import nli_metrics


def verifier_batch(data: dict[str, torch.Tensor], indices: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, ...]:
    source = data["source_ids"][indices].to(device=device, dtype=torch.long, non_blocking=True)
    lengths = data["source_lengths"][indices].to(device=device, dtype=torch.long, non_blocking=True)
    context_start = data["context_start"][indices].to(device=device, dtype=torch.long, non_blocking=True)
    positions = torch.arange(source.shape[1], device=device)[None]
    valid = positions < lengths[:, None]
    question = valid & (positions < context_start[:, None])
    start = data["candidate_start"][indices].to(device=device, dtype=torch.long, non_blocking=True)
    end = data["candidate_end"][indices].to(device=device, dtype=torch.long, non_blocking=True)
    candidate = valid & (positions >= start[:, None]) & (positions < end[:, None])
    labels = data["nli_label"][indices].to(device=device, dtype=torch.long, non_blocking=True)
    return source, valid, question, candidate, labels


@torch.inference_mode()
def evaluate_reader(model: NeedlePointerModel, data: dict[str, torch.Tensor], device: torch.device, batch_size: int, limit: int) -> dict[str, float]:
    model.eval()
    rows = min(len(data["source_ids"]), limit) if limit else len(data["source_ids"])
    total_tokens = total_copy = 0
    sequence = pointer = correct = 0.0
    for start in range(0, rows, batch_size):
        indices = torch.arange(start, min(start + batch_size, rows))
        source, source_valid, context_mask, decoder, target, valid, gold, _ = reader_batch(data, indices, device)
        loss = pointer_loss(model(source, source_valid, context_mask, decoder, valid), source, target, valid, gold)
        tokens = int(valid.sum())
        copied = int((valid & gold.ge(0)).sum())
        total_tokens += tokens
        total_copy += copied
        sequence += float(loss.sequence) * tokens
        pointer += float(loss.pointer_position) * copied
        correct += float(loss.pointer_accuracy) * copied
    model.train()
    return {
        "reader/sequence_nll": sequence / max(total_tokens, 1),
        "reader/pointer_position_nll": pointer / max(total_copy, 1),
        "reader/pointer_position_accuracy": correct / max(total_copy, 1),
        "reader/eval_rows": rows,
    }


@torch.inference_mode()
def evaluate_verifier(model: NeedlePointerModel, verifier: nn.Module, data: dict[str, torch.Tensor], device: torch.device, batch_size: int) -> dict[str, float]:
    model.eval()
    verifier.eval()
    logits, labels = [], []
    for start in range(0, len(data["source_ids"]), batch_size):
        indices = torch.arange(start, min(start + batch_size, len(data["source_ids"])))
        source, valid, question, candidate, target = verifier_batch(data, indices, device)
        logits.append(joint_verifier_logits(model, verifier, source, valid, question, candidate).float().cpu())
        labels.append(target.cpu())
    model.train()
    verifier.train()
    return nli_metrics(torch.cat(logits), torch.cat(labels))


def main() -> None:
    parser = argparse.ArgumentParser(description="Joint N2 reader + candidate verifier pilot with answer replay.")
    parser.add_argument("--reader-data-dir", type=Path, required=True)
    parser.add_argument("--verifier-data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--reader-eval-rows", type=int, default=1024)
    parser.add_argument("--reader-lr", type=float, default=1.0e-6)
    parser.add_argument("--head-lr", type=float, default=1.0e-4)
    parser.add_argument("--verification-ratio", type=float, default=0.5)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--run-name", default="needle26m-joint-reader-verifier")
    args = parser.parse_args()
    if not 0 < args.verification_ratio < 1:
        parser.error("--verification-ratio must be between zero and one")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    reader_train, reader_validation = load_split(args.reader_data_dir, "train"), load_split(args.reader_data_dir, "validation")
    verifier_train, verifier_validation = load_split(args.verifier_data_dir, "train"), load_split(args.verifier_data_dir, "validation")
    reader = NeedlePointerModel(NeedleConfig.public_checkpoint()).to(device=device, dtype=torch.bfloat16)
    load_start(reader, args.checkpoint, device)
    verifier = nn.Sequential(nn.Linear(reader.cfg.d_model * 4, args.hidden_dim), nn.GELU(), nn.Linear(args.hidden_dim, 3)).to(device=device, dtype=torch.bfloat16)
    counts = torch.bincount(verifier_train["nli_label"], minlength=3).float()
    class_weight = (len(verifier_train["nli_label"]) / (3 * counts.clamp_min(1))).to(device)
    optimizer = torch.optim.AdamW((
        {"params": reader.parameters(), "lr": args.reader_lr},
        {"params": verifier.parameters(), "lr": args.head_lr},
    ), betas=(0.9, 0.95), weight_decay=0.0)
    generator = torch.Generator().manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run = None
    if args.wandb:
        import wandb

        config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
        run = wandb.init(project="grounded-attn-qa", group="needle-rag", name=args.run_name, config=config)

    initial = {**evaluate_reader(reader, reader_validation, device, args.batch_size, args.reader_eval_rows), **evaluate_verifier(reader, verifier, verifier_validation, device, args.batch_size)}
    baseline_pointer = initial["reader/pointer_position_accuracy"]
    print(json.dumps({"step": 0, **initial}), flush=True)
    if run:
        run.log(initial, step=0)
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        if torch.rand((), generator=generator).item() < args.verification_ratio:
            indices = torch.randint(len(verifier_train["source_ids"]), (args.batch_size,), generator=generator)
            source, valid, question, candidate, labels = verifier_batch(verifier_train, indices, device)
            logits = joint_verifier_logits(reader, verifier, source, valid, question, candidate)
            loss = F.cross_entropy(logits.float(), labels, weight=class_weight)
            metrics = {"train/task": "verify", "train/verification_loss": float(loss.detach()), "train/verification_accuracy": float(logits.argmax(-1).eq(labels).float().mean())}
        else:
            indices = torch.randint(len(reader_train["source_ids"]), (args.batch_size,), generator=generator)
            source, source_valid, context_mask, decoder, target, valid, gold, _ = reader_batch(reader_train, indices, device)
            loss_info = pointer_loss(reader(source, source_valid, context_mask, decoder, valid), source, target, valid, gold)
            loss = loss_info.total
            metrics = {"train/task": "answer", "train/answer_loss": float(loss.detach()), "train/pointer_position_accuracy": float(loss_info.pointer_accuracy.detach())}
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_((*reader.parameters(), *verifier.parameters()), 1.0)
        optimizer.step()
        if step == 1 or step % 50 == 0:
            metrics.update({"train/grad_norm": float(grad_norm), "system/steps_per_second": step / (time.perf_counter() - started), "system/peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9})
            print(json.dumps({"step": step, **metrics}), flush=True)
            if run:
                run.log(metrics, step=step)
        if step % args.eval_every == 0 or step == args.steps:
            metrics = {**evaluate_reader(reader, reader_validation, device, args.batch_size, args.reader_eval_rows), **evaluate_verifier(reader, verifier, verifier_validation, device, args.batch_size)}
            metrics["reader/pointer_accuracy_delta"] = metrics["reader/pointer_position_accuracy"] - baseline_pointer
            print(json.dumps({"step": step, **metrics}), flush=True)
            if run:
                run.log(metrics, step=step)
            torch.save({"model": reader.state_dict(), "verifier": verifier.state_dict(), "step": step, "metrics": metrics, "args": vars(args)}, args.output_dir / f"step-{step:06d}.pt")
    if run:
        run.finish()


if __name__ == "__main__":
    main()
