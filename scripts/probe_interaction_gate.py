from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from grounded_qa.calibration import choose_threshold, sweep_thresholds
from grounded_qa.needle_pointer import NeedleAnswerablePointerModel
from grounded_qa.needleish import NeedleConfig
from scripts.analyze_pointer_confidence import binary_auc
from scripts.train_needle_n1 import load_split


def interaction_features(memory: torch.Tensor, source_valid: torch.Tensor, context_mask: torch.Tensor) -> torch.Tensor:
    """A readout probe: pooled question/context features with no reader updates."""
    question_mask = source_valid & ~context_mask
    question = (memory * question_mask[..., None]).sum(dim=1) / question_mask.sum(dim=1, keepdim=True).clamp_min(1)
    context = (memory * context_mask[..., None]).sum(dim=1) / context_mask.sum(dim=1, keepdim=True).clamp_min(1)
    return torch.cat((question, context, question * context, (question - context).abs()), dim=-1)


@torch.no_grad()
def encode_features(model, data: dict[str, torch.Tensor], indices: torch.Tensor, device: torch.device, batch_size: int) -> torch.Tensor:
    features = []
    for start in range(0, len(indices), batch_size):
        selected = indices[start : start + batch_size]
        source = data["source_ids"][selected].to(device=device, dtype=torch.long)
        lengths = data["source_lengths"][selected].to(device=device, dtype=torch.long)
        starts = data["context_start"][selected].to(device=device, dtype=torch.long)
        positions = torch.arange(source.shape[1], device=device)[None]
        valid = positions < lengths[:, None]
        context = (positions >= starts[:, None]) & valid
        features.append(interaction_features(model.encode(source, valid), valid, context).float().cpu())
    return torch.cat(features)


def safe_summary(scores: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    values, targets = scores.tolist(), labels.tolist()
    safe = choose_threshold(sweep_thresholds(values, targets), max_false_answer_rate=0.02)
    return {
        "auc": binary_auc(values, targets),
        "safe_threshold": safe.threshold,
        "safe_answer_coverage": safe.answer_coverage,
        "safe_false_answer_rate": safe.false_answer_rate,
    }


def parse_slice(value: str) -> tuple[str, int, int]:
    name, start, end = value.split(":")
    if int(start) < 0 or int(end) <= int(start):
        raise ValueError("slice boundaries must satisfy 0 <= start < end")
    return name, int(start), int(end)


def main() -> None:
    parser = argparse.ArgumentParser(description="Test whether frozen reader representations support a better answerability readout.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-rows", type=int, default=40_000)
    parser.add_argument("--feature-batch-size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3.0e-3)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--slice", action="append", default=[], help="Named validation range, e.g. official:123:456")
    args = parser.parse_args()

    train, validation = load_split(args.data_dir, "train"), load_split(args.data_dir, "validation")
    if "answerable" not in train or "answerable" not in validation:
        raise ValueError("interaction gate requires answerability labels")
    if not 0 < args.train_rows <= len(train["source_ids"]):
        raise ValueError("train rows must be between one and available training rows")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeedleAnswerablePointerModel(NeedleConfig.public_checkpoint()).to(
        device=device, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32
    )
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=False)["model"])
    model.eval()
    generator = torch.Generator().manual_seed(args.seed)
    train_indices = torch.randperm(len(train["source_ids"]), generator=generator)[: args.train_rows]
    validation_indices = torch.arange(len(validation["source_ids"]))
    train_features = encode_features(model, train, train_indices, device, args.feature_batch_size)
    validation_features = encode_features(model, validation, validation_indices, device, args.feature_batch_size)
    train_labels = train["answerable"][train_indices].float()
    validation_labels = validation["answerable"].bool()

    head = nn.Linear(train_features.shape[1], 1)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr)
    pos_weight = (len(train_labels) - train_labels.sum()) / train_labels.sum().clamp_min(1)
    for _ in range(args.steps):
        indices = torch.randint(len(train_features), (args.batch_size,), generator=generator)
        logits = head(train_features[indices]).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits, train_labels[indices], pos_weight=pos_weight)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    probabilities = head(validation_features).squeeze(-1).sigmoid().detach()
    summary = safe_summary(probabilities, validation_labels)
    slices = {}
    for value in args.slice:
        name, start, end = parse_slice(value)
        if end > len(probabilities):
            raise ValueError(f"slice {name} exceeds validation rows")
        values = probabilities[start:end]
        slices[name] = {
            "rows": len(values),
            "answerable": bool(validation_labels[start:end].all()),
            "median_probability": float(values.median()),
            "answer_rate_at_safe_threshold": float((values >= summary["safe_threshold"]).float().mean()),
        }
    report = {
        "checkpoint": str(args.checkpoint),
        "train_rows": args.train_rows,
        "features": "[question_mean, context_mean, product, absolute_difference]",
        "summary": summary,
        "slices": slices,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
