from __future__ import annotations

import argparse
import json
from pathlib import Path, PosixPath

import torch

from grounded_qa.negatives import REFUSAL
from grounded_qa.needle_pointer import NeedlePointerModel, answerability_interaction_features, candidate_verifier_head
from grounded_qa.needle_qa_data import _evidence_window
from grounded_qa.needle_tokenizer import NeedleTokenizer
from grounded_qa.needleish import NeedleConfig
from scripts.prepare_needle_n2 import SOURCE_LENGTH
from scripts.prepare_needle_n3_verifier import verifier_query


def candidate_offset(context: str, candidate: str) -> int:
    return context.lower().find(candidate.lower())


def is_answerable(row: dict) -> bool:
    return bool(row.get("answerable", row.get("condition") in {"correct", "counterfactual"}))


def load_head_state(path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    # Training checkpoints retain argparse Paths as harmless metadata.
    with torch.serialization.safe_globals([PosixPath]):
        return torch.load(path, map_location=device, weights_only=True)["head"]


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description="Gate reader outputs with a candidate-conditioned verifier head.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True, help="JSON report produced by evaluate_public_needle.py")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=0, help="Verifier hidden width; must match the head checkpoint.")
    args = parser.parse_args()

    report = json.loads(args.input.read_text())
    rows = report["examples"]
    tokenizer = NeedleTokenizer(args.tokenizer, append_markers=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = NeedlePointerModel(NeedleConfig.public_checkpoint()).to(device=device, dtype=dtype)
    model.load_backbone_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=False)["model"])
    model.eval()
    head = candidate_verifier_head(NeedleConfig.public_checkpoint().d_model, args.hidden_dim).to(device=device, dtype=dtype)
    head.load_state_dict(load_head_state(args.head, device))
    head.eval()

    eligible: list[tuple[int, list[int], int]] = []
    probabilities = [0.0] * len(rows)
    for index, row in enumerate(rows):
        candidate = row.get("raw_prediction", row.get("prediction", "")).strip()
        offset = candidate_offset(row["context"], candidate)
        if offset < 0:
            continue
        query = verifier_query(row["question"], candidate)
        window = _evidence_window(tokenizer, query, row["context"], offset, offset + len(candidate), max_source_length=SOURCE_LENGTH)
        if window is not None:
            eligible.append((index, window[0], window[3]))

    for start in range(0, len(eligible), args.batch_size):
        batch = eligible[start : start + args.batch_size]
        width = max(len(ids) for _, ids, _ in batch)
        source = torch.zeros((len(batch), width), dtype=torch.long, device=device)
        valid = torch.zeros_like(source, dtype=torch.bool)
        context = torch.zeros_like(source, dtype=torch.bool)
        for row_index, (output_index, ids, context_start) in enumerate(batch):
            source[row_index, : len(ids)] = torch.tensor(ids, device=device)
            valid[row_index, : len(ids)] = True
            context[row_index, context_start : len(ids)] = True
        memory = model.encode(source, valid)
        scores = head(answerability_interaction_features(memory, valid, context)).squeeze(-1).sigmoid().float().cpu().tolist()
        for (output_index, _, _), score in zip(batch, scores):
            probabilities[output_index] = score

    verified = []
    for row, probability in zip(rows, probabilities):
        raw = row.get("raw_prediction", row.get("prediction", "")).strip()
        accepted = probability >= args.threshold
        verified.append({
            **row,
            "candidate_probability": probability,
            "candidate_accepted": accepted,
            "prediction": raw if accepted else REFUSAL,
        })
    answerable = [row for row in verified if is_answerable(row)]
    unsupported = [row for row in verified if not is_answerable(row)]
    summary = {
        "rows": len(verified),
        "accepted": sum(row["candidate_accepted"] for row in verified),
        "false_refusal_rate": sum(not row["candidate_accepted"] for row in answerable) / max(len(answerable), 1),
        "false_answer_rate": sum(row["candidate_accepted"] for row in unsupported) / max(len(unsupported), 1),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"input": str(args.input), "threshold": args.threshold, "summary": summary, "examples": verified}, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
