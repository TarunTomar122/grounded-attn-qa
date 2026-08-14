from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from grounded_qa.metrics import exact_match, token_f1
from grounded_qa.needle_tokenizer import EOS_ID, NeedleTokenizer
from grounded_qa.needleish import NeedleConfig, NeedleishModel, load_public_checkpoint


def newton_schulz(gradient: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Needle's Newton-Schulz polar approximation for a 2D dense gradient."""
    a, b, c = 3.4445, -4.7750, 2.0315
    dtype = gradient.dtype
    x = gradient.float() / (gradient.float().norm() + 1.0e-7)
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    for _ in range(steps):
        gram = x @ x.T
        x = a * x + (b * gram + c * gram @ gram) @ x
    return x.T.to(dtype) if transposed else x.to(dtype)


class Muon(torch.optim.Optimizer):
    """Minimal PyTorch equivalent of Needle's dense-kernel Muon transform."""

    def __init__(self, params, lr: float, momentum: float = 0.95, weight_decay: float = 0.01):
        super().__init__(params, {"lr": lr, "momentum": momentum, "weight_decay": weight_decay})

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure else None
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                if parameter.grad.ndim != 2:
                    raise ValueError("Muon only supports 2D dense weights")
                update = newton_schulz(parameter.grad)
                momentum = self.state[parameter].setdefault("momentum_buffer", torch.zeros_like(update))
                momentum.mul_(group["momentum"]).add_(update)
                update.add_(momentum, alpha=group["momentum"])
                update.add_(parameter, alpha=group["weight_decay"])
                parameter.add_(update, alpha=-group["lr"])
        return loss


def optimizers_for(model: nn.Module, adam_lr: float, muon_lr: float) -> dict[str, torch.optim.Optimizer]:
    if not muon_lr:
        return {"adam": torch.optim.AdamW(model.parameters(), lr=adam_lr, betas=(0.9, 0.95), weight_decay=0.0)}
    dense_ids = {id(module.weight) for module in model.modules() if isinstance(module, nn.Linear)}
    dense = [parameter for parameter in model.parameters() if id(parameter) in dense_ids]
    other = [parameter for parameter in model.parameters() if id(parameter) not in dense_ids]
    return {
        "adam": torch.optim.AdamW(other, lr=adam_lr, betas=(0.9, 0.95), weight_decay=0.0),
        "muon": Muon(dense, lr=muon_lr),
    }


def load_initial_checkpoint(model: nn.Module, path: Path, device: torch.device) -> None:
    if path.suffix == ".safetensors":
        load_public_checkpoint(model, path)
    else:
        model.load_state_dict(torch.load(path, map_location=device, weights_only=False)["model"])


def load_split(data_dir: Path, split: str) -> dict[str, torch.Tensor]:
    files = sorted(data_dir.glob(f"*-{split}.pt"))
    if not files:
        raise FileNotFoundError(f"No {split} tensors in {data_dir}")
    shards = [torch.load(path, map_location="cpu", weights_only=True) for path in files]
    return {key: torch.cat([shard[key] for shard in shards]) for key in shards[0]}


def batch(data: dict[str, torch.Tensor], indices: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, ...]:
    source = data["source_ids"][indices].to(device=device, dtype=torch.long, non_blocking=True)
    target = data["target_ids"][indices].to(device=device, dtype=torch.long, non_blocking=True)
    source_lengths = data["source_lengths"][indices].to(device=device, dtype=torch.long, non_blocking=True)
    target_lengths = data["target_lengths"][indices].to(device=device, dtype=torch.long, non_blocking=True)
    source_valid = torch.arange(source.shape[1], device=device)[None] < source_lengths[:, None]
    target_valid = torch.arange(target.shape[1], device=device)[None] < target_lengths[:, None]
    decoder = torch.zeros_like(target)
    decoder[:, 0] = EOS_ID
    decoder[:, 1:] = target[:, :-1]
    return source, source_valid, decoder, target, target_valid


def losses(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    z_weight: float,
    first_token_weight: float,
    eos_token_weight: float,
) -> tuple[torch.Tensor, ...]:
    selected_logits = logits[valid].float()
    selected_target = target[valid]
    per_token_ce = F.cross_entropy(selected_logits, selected_target, reduction="none")
    first_mask = valid.nonzero()[:, 1].eq(0)
    eos_mask = selected_target.eq(EOS_ID)
    weights = 1 + first_mask * (first_token_weight - 1) + eos_mask * (eos_token_weight - 1)
    ce = (per_token_ce * weights).sum() / weights.sum()
    z = selected_logits.logsumexp(dim=-1).square().mean()
    correct = selected_logits.argmax(dim=-1).eq(selected_target)
    accuracy = correct.float().mean()
    first_ce = per_token_ce[first_mask].mean()
    first_accuracy = correct[first_mask].float().mean()
    eos_accuracy = correct[eos_mask].float().mean()
    return ce + z_weight * z, ce, z, accuracy, first_ce, first_accuracy, eos_accuracy


def lr_at(step: int, total_steps: int, peak: float) -> float:
    warmup = max(1, round(total_steps * 0.05))
    decay = max(1, round(total_steps * 0.15))
    if step < warmup:
        return peak * (step + 1) / warmup
    if step < total_steps - decay:
        return peak
    progress = (step - (total_steps - decay)) / decay
    return peak * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * progress)))


@torch.inference_mode()
def evaluate(model, data: dict[str, torch.Tensor], device: torch.device, batch_size: int, z_weight: float, first_token_weight: float, eos_token_weight: float) -> dict[str, float]:
    model.eval()
    totals = {"tokens": 0, "weighted_tokens": 0.0, "rows": 0, "ce": 0.0, "z": 0.0, "correct": 0.0, "first_ce": 0.0, "first_correct": 0.0, "eos_correct": 0.0, "wrong_ce": 0.0}
    for start in range(0, len(data["source_ids"]), batch_size):
        indices = torch.arange(start, min(start + batch_size, len(data["source_ids"])))
        source, source_valid, decoder, target, valid = batch(data, indices, device)
        logits = model(source, source_valid, decoder, valid)
        total, ce, z, accuracy, first_ce, first_accuracy, eos_accuracy = losses(logits, target, valid, z_weight, first_token_weight, eos_token_weight)
        wrong_logits = model(source.roll(1, 0), source_valid.roll(1, 0), decoder, valid)
        _, wrong_ce, *_ = losses(wrong_logits, target, valid, 0.0, first_token_weight, eos_token_weight)
        tokens = int(valid.sum())
        rows = len(indices)
        weighted_tokens = tokens + (first_token_weight + eos_token_weight - 2) * rows
        totals["tokens"] += tokens
        totals["weighted_tokens"] += weighted_tokens
        totals["rows"] += rows
        totals["ce"] += float(ce) * weighted_tokens
        totals["z"] += float(z) * tokens
        totals["correct"] += float(accuracy) * tokens
        totals["first_ce"] += float(first_ce) * rows
        totals["first_correct"] += float(first_accuracy) * rows
        totals["eos_correct"] += float(eos_accuracy) * rows
        totals["wrong_ce"] += float(wrong_ce) * weighted_tokens
    tokens = max(totals["tokens"], 1)
    weighted_tokens = max(totals["weighted_tokens"], 1)
    ce = totals["ce"] / weighted_tokens
    z = totals["z"] / tokens
    wrong_ce = totals["wrong_ce"] / weighted_tokens
    model.train()
    return {
        "val/loss": ce + z_weight * z,
        "val/ce": ce,
        "val/z": z,
        "val/token_accuracy": totals["correct"] / tokens,
        "val/first_token_ce": totals["first_ce"] / totals["rows"],
        "val/first_token_accuracy": totals["first_correct"] / totals["rows"],
        "val/eos_accuracy": totals["eos_correct"] / totals["rows"],
        "val/wrong_context_ce": wrong_ce,
        "val/context_dependency_gap": wrong_ce - ce,
    }


@torch.inference_mode()
def probe(model: NeedleishModel, tokenizer: NeedleTokenizer, data: dict[str, torch.Tensor], device: torch.device, count: int = 8) -> tuple[dict[str, float], list[dict]]:
    model.eval()
    indices = torch.arange(min(count, len(data["source_ids"])))
    source, source_valid, _, target, _ = batch(data, indices, device)
    memory = model.encode(source, source_valid)
    decoder = torch.full((len(indices), 1), EOS_ID, dtype=torch.long, device=device)
    finished = torch.zeros(len(indices), dtype=torch.bool, device=device)
    for _ in range(512):
        logits = model.decode(decoder, memory, source_valid, torch.ones_like(decoder, dtype=torch.bool))
        next_ids = logits[:, -1].argmax(dim=-1)
        next_ids = torch.where(finished, torch.full_like(next_ids, EOS_ID), next_ids)
        decoder = torch.cat((decoder, next_ids[:, None]), dim=1)
        finished |= next_ids.eq(EOS_ID)
        if finished.all():
            break
    rows = []
    for generated, gold in zip(decoder[:, 1:].cpu().tolist(), target.cpu().tolist()):
        emitted_eos = EOS_ID in generated
        if EOS_ID in generated:
            generated = generated[: generated.index(EOS_ID)]
        gold = gold[: gold.index(EOS_ID)] if EOS_ID in gold else [token for token in gold if token]
        prediction = tokenizer.decode(generated).strip()
        answer = tokenizer.decode(gold).strip()
        rows.append({"prediction": prediction, "gold": answer, "eos": emitted_eos, "tokens": len(generated)})
    model.train()
    return {
        "probe/em": sum(exact_match(row["prediction"], row["gold"]) for row in rows) / len(rows),
        "probe/token_f1": sum(token_f1(row["prediction"], row["gold"]) for row in rows) / len(rows),
        "probe/eos_rate": sum(row["eos"] for row in rows) / len(rows),
        "probe/mean_tokens": sum(row["tokens"] for row in rows) / len(rows),
    }, rows


def save_checkpoint(path: Path, model, optimizers: dict[str, torch.optim.Optimizer], step: int, seen_tokens: int, epoch: int, offset: int, args: argparse.Namespace) -> None:
    eager = getattr(model, "_orig_mod", model)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip()
    torch.save({
        "model": eager.state_dict(),
        "optimizers": {name: optimizer.state_dict() for name, optimizer in optimizers.items()},
        "step": step,
        "seen_tokens": seen_tokens,
        "epoch": epoch,
        "offset": offset,
        "config": NeedleConfig.public_checkpoint().to_dict(),
        "args": vars(args),
        "git_commit": commit,
    }, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="N1 continued training on answerable SYNTH RAG.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--muon-lr", type=float, default=0.0)
    parser.add_argument("--z-loss", type=float, default=1e-4)
    parser.add_argument("--first-token-weight", type=float, default=20.0)
    parser.add_argument("--eos-token-weight", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--run-name")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    train = load_split(args.data_dir, "train")
    validation = load_split(args.data_dir, "validation")
    eager = NeedleishModel(NeedleConfig.public_checkpoint()).to(device=device, dtype=torch.bfloat16)
    load_initial_checkpoint(eager, args.checkpoint, device)
    model = torch.compile(eager) if args.compile else eager
    optimizers = optimizers_for(eager, args.lr, args.muon_lr)
    if args.weight_decay:
        for group in optimizers["adam"].param_groups:
            group["weight_decay"] = args.weight_decay
    step = 0
    seen_tokens = 0
    epoch = 0
    offset = 0
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        eager.load_state_dict(state["model"])
        optimizer_states = state["optimizers"] if "optimizers" in state else {"adam": state["optimizer"]}
        for name, optimizer_state in optimizer_states.items():
            optimizers[name].load_state_dict(optimizer_state)
        step = state["step"]
        seen_tokens = state.get("seen_tokens", 0)
        epoch = state.get("epoch", 0)
        offset = state.get("offset", 0)
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
        default_name = f"needle26m-n1-adam{args.lr:g}-muon{args.muon_lr:g}-first{args.first_token_weight:g}-eos{args.eos_token_weight:g}"
        run = wandb.init(project="grounded-attn-qa", group="needle-rag", name=args.run_name or default_name, config={**config, "train_rows": len(train["source_ids"]), "validation_rows": len(validation["source_ids"]), "total_steps": total_steps})

    initial = evaluate(model, validation, device, args.batch_size, args.z_loss, args.first_token_weight, args.eos_token_weight)
    probe_metrics, probe_rows = probe(eager, NeedleTokenizer(args.tokenizer, append_markers=False), validation, device)
    print(json.dumps({"step": step, **initial, **probe_metrics, "examples": probe_rows[:2]}, ensure_ascii=False), flush=True)
    if run:
        run.log({**initial, **probe_metrics}, step=step)
        run.log({"probe/examples": wandb.Table(columns=["prediction", "gold", "eos", "tokens"], data=[[row[key] for key in ("prediction", "gold", "eos", "tokens")] for row in probe_rows])}, step=step)
    started = time.perf_counter()
    last_log = started
    last_log_step = step
    last_log_tokens = seen_tokens
    while step < total_steps:
        order = torch.randperm(len(train["source_ids"]), generator=torch.Generator().manual_seed(args.seed + epoch))
        for start in range(offset, epoch_steps * args.batch_size, args.batch_size):
            if step >= total_steps:
                break
            indices = order[start : start + args.batch_size]
            source, source_valid, decoder, target, valid = batch(train, indices, device)
            adam_lr = lr_at(step, total_steps, args.lr)
            muon_lr = lr_at(step, total_steps, args.muon_lr)
            for group in optimizers["adam"].param_groups:
                group["lr"] = adam_lr
            if "muon" in optimizers:
                for group in optimizers["muon"].param_groups:
                    group["lr"] = muon_lr
            for optimizer in optimizers.values():
                optimizer.zero_grad(set_to_none=True)
            logits = model(source, source_valid, decoder, valid)
            loss, ce, z, accuracy, first_ce, first_accuracy, eos_accuracy = losses(logits, target, valid, args.z_loss, args.first_token_weight, args.eos_token_weight)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            for optimizer in optimizers.values():
                optimizer.step()
            step += 1
            offset = start + args.batch_size
            seen_tokens += int(source_valid.sum() + valid.sum())
            if step == 1 or step % 50 == 0:
                elapsed = time.perf_counter() - started
                interval = time.perf_counter() - last_log
                metrics = {
                    "train/loss": float(loss.detach()),
                    "train/ce": float(ce.detach()),
                    "train/z": float(z.detach()),
                    "train/token_accuracy": float(accuracy.detach()),
                    "train/first_token_ce": float(first_ce.detach()),
                    "train/first_token_accuracy": float(first_accuracy.detach()),
                    "train/eos_accuracy": float(eos_accuracy.detach()),
                    "train/grad_norm": float(grad_norm),
                    "train/adam_lr": adam_lr,
                    "train/muon_lr": muon_lr,
                    "system/actual_tokens_per_second": (seen_tokens - last_log_tokens) / max(interval, 1e-6),
                    "system/padded_tokens_per_second": ((step - last_log_step) * args.batch_size * 1536) / max(interval, 1e-6),
                    "system/run_seconds": elapsed,
                    "system/peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9,
                }
                print(json.dumps({"step": step, **metrics}), flush=True)
                if run:
                    run.log(metrics, step=step)
                last_log = time.perf_counter()
                last_log_step = step
                last_log_tokens = seen_tokens
            if step % args.eval_every == 0 or step == total_steps:
                metrics = evaluate(model, validation, device, args.batch_size, args.z_loss, args.first_token_weight, args.eos_token_weight)
                probe_metrics, probe_rows = probe(eager, NeedleTokenizer(args.tokenizer, append_markers=False), validation, device)
                print(json.dumps({"step": step, **metrics, **probe_metrics, "examples": probe_rows[:2]}, ensure_ascii=False), flush=True)
                if run:
                    run.log({**metrics, **probe_metrics}, step=step)
                    run.log({"probe/examples": wandb.Table(columns=["prediction", "gold", "eos", "tokens"], data=[[row[key] for key in ("prediction", "gold", "eos", "tokens")] for row in probe_rows])}, step=step)
                last_log = time.perf_counter()
                last_log_step = step
                last_log_tokens = seen_tokens
            if step % args.checkpoint_every == 0 or step == total_steps:
                save_checkpoint(args.output_dir / f"step-{step:06d}.pt", model, optimizers, step, seen_tokens, epoch, offset, args)
        epoch += 1
        offset = 0
    if run:
        run.finish()


if __name__ == "__main__":
    main()
