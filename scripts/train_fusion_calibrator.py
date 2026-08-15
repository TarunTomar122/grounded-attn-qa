"""Fit a tiny correctness calibrator over reader-visible and verifier signals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from scripts.analyze_pointer_confidence import binary_auc
from scripts.evaluate_candidate_verifier import safe_gate_summary

FEATURE_NAMES = ("verifier_support", "generated_tokens", "eos", "unsupported_number", "unsupported_entity", "literal_candidate")


def feature_rows(rows: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
    features, labels = [], []
    for row in rows:
        candidate = str(row.get("raw_prediction", row.get("prediction", ""))).strip().lower()
        literal = bool(candidate) and candidate in row["context"].lower()
        features.append((
            float(row["candidate_probability"]),
            float(row.get("generated_tokens", 0)),
            float(bool(row.get("eos", False))),
            float(row.get("unsupported_number_rate", 0.0)),
            float(row.get("unsupported_entity_rate", 0.0)),
            float(literal),
        ))
        labels.append(float(float(row.get("em", 0.0)) == 1.0))
    return torch.tensor(features, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)


def risk_curve(rows: list[dict], probabilities: torch.Tensor, points: int = 101) -> list[dict]:
    curve = []
    for index in range(points):
        threshold = index / (points - 1)
        gated = [{**row, "candidate_accepted": float(probability) >= threshold} for row, probability in zip(rows, probabilities)]
        curve.append({"threshold": threshold, **safe_gate_summary(gated)})
    return curve


@torch.inference_mode()
def metrics(rows: list[dict], features: torch.Tensor, labels: torch.Tensor, mean: torch.Tensor, scale: torch.Tensor, model: nn.Module) -> dict:
    probabilities = model((features - mean) / scale).sigmoid().squeeze(-1)
    curve = risk_curve(rows, probabilities)
    safe = [point for point in curve if point["accepted_answer_risk"] <= 0.02]
    best = max(safe, key=lambda point: point["safe_answer_coverage"])
    return {"auc": binary_auc(probabilities.tolist(), labels.bool().tolist()), "best_at_2pct_risk": best, "curve": curve}


def load_rows(path: Path) -> list[dict]:
    return json.loads(path.read_text())["examples"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a small reader-plus-verifier correctness calibrator.")
    parser.add_argument("--train", type=Path, required=True, help="Scored candidate-verifier JSON on context-disjoint train paragraphs.")
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1_000)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=47)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    train_rows, validation_rows = load_rows(args.train), load_rows(args.validation)
    train_x, train_y = feature_rows(train_rows)
    validation_x, validation_y = feature_rows(validation_rows)
    mean, scale = train_x.mean(0), train_x.std(0).clamp_min(1e-6)
    model = nn.Linear(len(FEATURE_NAMES), 1)
    positive_weight = torch.tensor([(len(train_y) - train_y.sum()) / train_y.sum().clamp_min(1)])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    for _ in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(model((train_x - mean) / scale).squeeze(-1), train_y, pos_weight=positive_weight)
        loss.backward()
        optimizer.step()

    report = {
        "features": FEATURE_NAMES,
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "train": metrics(train_rows, train_x, train_y, mean, scale, model),
        "validation": metrics(validation_rows, validation_x, validation_y, mean, scale, model),
        "state": {"weight": model.weight.detach().tolist(), "bias": model.bias.detach().tolist(), "mean": mean.tolist(), "scale": scale.tolist()},
        "args": vars(args),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({"train": report["train"]["best_at_2pct_risk"], "validation": report["validation"]["best_at_2pct_risk"], "validation_auc": report["validation"]["auc"]}, indent=2))


if __name__ == "__main__":
    main()
