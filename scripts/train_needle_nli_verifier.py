from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from grounded_qa.calibration import choose_threshold, sweep_thresholds
from grounded_qa.needle_pointer import NeedlePointerModel, candidate_span_features
from grounded_qa.needle_verifier import NeedleVerifierAdapter
from grounded_qa.needleish import NeedleConfig
from scripts.analyze_pointer_confidence import binary_auc
from scripts.train_needle_n1 import load_split


def nli_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    probabilities = logits.softmax(dim=-1)[:, 2].float()
    support = labels.eq(2)
    safe = choose_threshold(sweep_thresholds(probabilities.tolist(), support.tolist()), max_false_answer_rate=0.02)
    return {
        "val/nli_accuracy": float(logits.argmax(dim=-1).eq(labels).float().mean()),
        "val/support_auc": binary_auc(probabilities.tolist(), support.tolist()),
        "val/safe_threshold": safe.threshold,
        "val/safe_answer_coverage": safe.answer_coverage,
        "val/safe_false_answer_rate": safe.false_answer_rate,
    }


def scores(model: nn.Module, verifier: nn.Module, data: dict[str, torch.Tensor], indices: torch.Tensor, device: torch.device, *, decoder_verifier: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    source = data["source_ids"][indices].to(device=device, dtype=torch.long)
    lengths = data["source_lengths"][indices].to(device=device, dtype=torch.long)
    context_start = data["context_start"][indices].to(device=device, dtype=torch.long)
    positions = torch.arange(source.shape[1], device=device)[None]
    valid = positions < lengths[:, None]
    question = valid & (positions < context_start[:, None])
    if "evidence_start" in data:
        evidence_start = data["evidence_start"][indices].to(device=device, dtype=torch.long)
        evidence_end = data["evidence_end"][indices].to(device=device, dtype=torch.long)
        candidate = valid & (positions >= evidence_start[:, None]) & (positions < evidence_end[:, None])
    elif "candidate_start" in data:
        candidate_start = data["candidate_start"][indices].to(device=device, dtype=torch.long)
        candidate_end = data["candidate_end"][indices].to(device=device, dtype=torch.long)
        candidate = valid & (positions >= candidate_start[:, None]) & (positions < candidate_end[:, None])
    else:
        candidate = valid & ~question
    features = model.verify(source, valid, question | candidate) if decoder_verifier else candidate_span_features(model.encode(source, valid), valid, question, candidate)
    return verifier(features), data["nli_label"][indices].to(device=device, dtype=torch.long)


@torch.inference_mode()
def evaluate(model: nn.Module, verifier: nn.Module, data: dict[str, torch.Tensor], device: torch.device, batch_size: int, *, decoder_verifier: bool = False) -> dict[str, float]:
    logits, labels = [], []
    for start in range(0, len(data["source_ids"]), batch_size):
        prediction, target = scores(model, verifier, data, torch.arange(start, min(start + batch_size, len(data["source_ids"]))), device, decoder_verifier=decoder_verifier)
        logits.append(prediction.float().cpu())
        labels.append(target.cpu())
    return nli_metrics(torch.cat(logits), torch.cat(labels))


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a separate N2-initialized support/refute/neutral verifier.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--replay-data-dir", type=Path, help="Optional equally sampled second verifier format for adapter specialization.")
    parser.add_argument("--validation-data-dir", type=Path, help="Optional deployment-format validation split.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1.0e-5)
    parser.add_argument("--head-lr", type=float, default=1.0e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--adapter-rank", type=int, default=0, help="Enable isolated verifier adapters; zero reproduces the legacy full-backbone pilot.")
    parser.add_argument("--decoder-verifier", action="store_true", help="Classify a decoder cross-attended verification token rather than pooled encoder features.")
    parser.add_argument("--init", type=Path, help="Optional earlier adapter/head checkpoint, for NLI-to-QA specialization.")
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--run-name", default="needle26m-nli-verifier")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    train = load_split(args.data_dir, "train")
    replay_train = load_split(args.replay_data_dir, "train") if args.replay_data_dir else None
    validation = load_split(args.validation_data_dir or args.data_dir, "validation")
    if "nli_label" not in train or "nli_label" not in validation:
        raise ValueError("NLI verifier data requires support/refute/neutral labels")
    reader = NeedlePointerModel(NeedleConfig.public_checkpoint()).to(device=device, dtype=torch.bfloat16)
    reader.load_backbone_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=False)["model"])
    if args.decoder_verifier and not args.adapter_rank:
        raise ValueError("decoder verifier requires adapter-rank")
    model: nn.Module = NeedleVerifierAdapter(reader, args.adapter_rank, decoder=args.decoder_verifier).to(device=device, dtype=torch.bfloat16) if args.adapter_rank else reader
    verifier = nn.Sequential(
        nn.Linear(NeedleConfig.public_checkpoint().d_model if args.decoder_verifier else NeedleConfig.public_checkpoint().d_model * 4, args.hidden_dim), nn.GELU(), nn.Linear(args.hidden_dim, 3)
    ).to(device=device, dtype=torch.bfloat16)
    if args.init:
        state = torch.load(args.init, map_location=device, weights_only=False)
        verifier.load_state_dict(state["verifier"])
        if args.adapter_rank:
            model.adapters.load_state_dict(state["adapter"])  # type: ignore[union-attr]
            if args.decoder_verifier:
                model.decoder_adapters.load_state_dict(state["decoder_adapter"])  # type: ignore[union-attr]
        elif "model" in state:
            model.load_state_dict(state["model"])
    counts = torch.bincount(train["nli_label"], minlength=3).float()
    class_weight = (len(train["nli_label"]) / (3 * counts.clamp_min(1))).to(device)
    optimizer = torch.optim.AdamW((
        {"params": model.parameters(), "lr": args.lr},
        {"params": verifier.parameters(), "lr": args.head_lr},
    ), betas=(0.9, 0.95), weight_decay=0.0)
    generator = torch.Generator().manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run = None
    if args.wandb:
        import wandb

        run = wandb.init(project="grounded-attn-qa", group="needle-rag", name=args.run_name, config={**vars(args), "train_rows": len(train["source_ids"]), "validation_rows": len(validation["source_ids"]), "class_counts": counts.tolist()})
    model.eval()
    initial = evaluate(model, verifier, validation, device, args.batch_size, decoder_verifier=args.decoder_verifier)
    print(json.dumps({"step": 0, **initial}), flush=True)
    if run:
        run.log(initial, step=0)
    model.train()
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        active_train = replay_train if replay_train is not None and torch.rand((), generator=generator).item() < 0.5 else train
        indices = torch.randint(len(active_train["source_ids"]), (args.batch_size,), generator=generator)
        logits, labels = scores(model, verifier, active_train, indices, device, decoder_verifier=args.decoder_verifier)
        loss = F.cross_entropy(logits.float(), labels, weight=class_weight)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_((*model.parameters(), *verifier.parameters()), 1.0)
        optimizer.step()
        if step == 1 or step % 50 == 0:
            metrics = {"train/loss": float(loss.detach()), "train/accuracy": float(logits.argmax(dim=-1).eq(labels).float().mean()), "train/grad_norm": float(grad_norm), "system/steps_per_second": step / (time.perf_counter() - started)}
            print(json.dumps({"step": step, **metrics}), flush=True)
            if run:
                run.log(metrics, step=step)
        if step % args.eval_every == 0 or step == args.steps:
            model.eval()
            metrics = evaluate(model, verifier, validation, device, args.batch_size, decoder_verifier=args.decoder_verifier)
            print(json.dumps({"step": step, **metrics}), flush=True)
            if run:
                run.log(metrics, step=step)
            model.train()
            checkpoint = {"verifier": verifier.state_dict(), "step": step, "args": vars(args)}
            if args.adapter_rank:
                checkpoint["adapter"] = model.adapters.state_dict()  # type: ignore[union-attr]
                if args.decoder_verifier:
                    checkpoint["decoder_adapter"] = model.decoder_adapters.state_dict()  # type: ignore[union-attr]
            else:
                checkpoint["model"] = model.state_dict()
            torch.save(checkpoint, args.output_dir / f"step-{step:06d}.pt")
    if run:
        run.finish()


if __name__ == "__main__":
    main()
