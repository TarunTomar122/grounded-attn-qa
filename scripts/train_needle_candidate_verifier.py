from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from grounded_qa.calibration import choose_threshold, sweep_thresholds
from grounded_qa.needle_pointer import NeedlePointerModel, answerability_interaction_features
from grounded_qa.needleish import NeedleConfig
from scripts.analyze_pointer_confidence import binary_auc
from scripts.train_needle_n1 import load_split


def scores(model: NeedlePointerModel, head: nn.Linear, data: dict[str, torch.Tensor], indices: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    source = data["source_ids"][indices].to(device=device, dtype=torch.long)
    lengths = data["source_lengths"][indices].to(device=device, dtype=torch.long)
    starts = data["context_start"][indices].to(device=device, dtype=torch.long)
    positions = torch.arange(source.shape[1], device=device)[None]
    valid = positions < lengths[:, None]
    context = valid & (positions >= starts[:, None])
    return head(answerability_interaction_features(model.encode(source, valid), valid, context)).squeeze(-1), data["answerable"][indices].float().to(device)


@torch.inference_mode()
def evaluate(model: NeedlePointerModel, head: nn.Linear, data: dict[str, torch.Tensor], device: torch.device, batch_size: int) -> dict[str, float]:
    probabilities, labels = [], []
    for start in range(0, len(data["source_ids"]), batch_size):
        logits, target = scores(model, head, data, torch.arange(start, min(start + batch_size, len(data["source_ids"]))), device)
        probabilities.append(logits.sigmoid().float().cpu())
        labels.append(target.bool().cpu())
    values, targets = torch.cat(probabilities), torch.cat(labels)
    safe = choose_threshold(sweep_thresholds(values.tolist(), targets.tolist()), max_false_answer_rate=0.02)
    return {
        "val/bce": float(F.binary_cross_entropy(values, targets.float())),
        "val/auc": binary_auc(values.tolist(), targets.tolist()),
        "val/safe_threshold": safe.threshold,
        "val/safe_answer_coverage": safe.answer_coverage,
        "val/safe_false_answer_rate": safe.false_answer_rate,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune Needle encoder for candidate-conditioned evidence verification.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--run-name", default="needle26m-candidate-verifier")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    train, validation = load_split(args.data_dir, "train"), load_split(args.data_dir, "validation")
    model = NeedlePointerModel(NeedleConfig.public_checkpoint()).to(device=device, dtype=torch.bfloat16)
    model.load_backbone_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=False)["model"])
    head = nn.Linear(NeedleConfig.public_checkpoint().d_model * 4, 1).to(device=device, dtype=torch.bfloat16)
    nn.init.zeros_(head.weight)
    nn.init.zeros_(head.bias)
    optimizer = torch.optim.AdamW((
        {"params": model.parameters(), "lr": args.lr},
        {"params": head.parameters(), "lr": args.head_lr},
    ), betas=(0.9, 0.95), weight_decay=0.0)
    generator = torch.Generator().manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run = None
    if args.wandb:
        import wandb

        run = wandb.init(project="grounded-attn-qa", group="needle-rag", name=args.run_name, config={**vars(args), "train_rows": len(train["source_ids"]), "validation_rows": len(validation["source_ids"])})
    model.eval()
    initial = evaluate(model, head, validation, device, args.batch_size)
    print(json.dumps({"step": 0, **initial}), flush=True)
    if run:
        run.log(initial, step=0)
    model.train()
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        indices = torch.randint(len(train["source_ids"]), (args.batch_size,), generator=generator)
        logits, labels = scores(model, head, train, indices, device)
        loss = F.binary_cross_entropy_with_logits(logits.float(), labels, pos_weight=torch.tensor(2.0, device=device))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_((*model.parameters(), *head.parameters()), 1.0)
        optimizer.step()
        if step == 1 or step % 50 == 0:
            metrics = {"train/loss": float(loss.detach()), "train/accuracy": float((logits.ge(0) == labels.bool()).float().mean()), "train/grad_norm": float(grad_norm), "system/steps_per_second": step / (time.perf_counter() - started)}
            print(json.dumps({"step": step, **metrics}), flush=True)
            if run:
                run.log(metrics, step=step)
        if step % args.eval_every == 0 or step == args.steps:
            model.eval()
            metrics = evaluate(model, head, validation, device, args.batch_size)
            print(json.dumps({"step": step, **metrics}), flush=True)
            if run:
                run.log(metrics, step=step)
            model.train()
            torch.save({"model": model.state_dict(), "head": head.state_dict(), "step": step, "args": vars(args)}, args.output_dir / f"step-{step:06d}.pt")
    if run:
        run.finish()


if __name__ == "__main__":
    main()
