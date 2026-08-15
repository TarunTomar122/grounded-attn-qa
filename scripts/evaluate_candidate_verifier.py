from __future__ import annotations

import argparse
import json
from pathlib import Path, PosixPath

import torch
from torch import nn

from grounded_qa.negatives import REFUSAL
from grounded_qa.needle_pointer import NeedlePointerModel, answerability_interaction_features, candidate_span_features, candidate_verifier_head
from grounded_qa.needle_verifier import NeedleVerifierAdapter
from grounded_qa.needle_qa_data import _evidence_window
from grounded_qa.needle_tokenizer import NeedleTokenizer
from grounded_qa.needleish import NeedleConfig
from scripts.prepare_needle_n2 import SOURCE_LENGTH
from scripts.prepare_needle_n3_verifier import verifier_claim, verifier_query


def candidate_offset(context: str, candidate: str) -> int:
    return context.lower().find(candidate.lower())


def is_answerable(row: dict) -> bool:
    return bool(row.get("answerable", row.get("condition") in {"correct", "counterfactual"}))


def safe_gate_summary(rows: list[dict]) -> dict[str, float | int]:
    """Measure a gate by retained correct answers and accepted wrong answers."""
    answerable = [row for row in rows if is_answerable(row)]
    raw_correct = [row for row in rows if float(row.get("em", 0.0)) == 1.0]
    raw_wrong = [row for row in rows if float(row.get("em", 0.0)) != 1.0]
    accepted = [row for row in rows if row["candidate_accepted"]]
    accepted_correct = [row for row in accepted if float(row.get("em", 0.0)) == 1.0]
    accepted_wrong = [row for row in accepted if float(row.get("em", 0.0)) != 1.0]
    return {
        "rows": len(rows),
        "accepted": len(accepted),
        "answer_coverage": sum(row["candidate_accepted"] for row in answerable) / max(len(answerable), 1),
        "safe_answer_coverage": len(accepted_correct) / max(len(answerable), 1),
        "accepted_answer_risk": len(accepted_wrong) / max(len(accepted), 1),
        "wrong_reader_answer_accept_rate": len(accepted_wrong) / max(len(raw_wrong), 1),
        "correct_reader_answer_refusal_rate": sum(not row["candidate_accepted"] for row in raw_correct) / max(len(raw_correct), 1),
        "false_refusal_rate": sum(not row["candidate_accepted"] for row in answerable) / max(len(answerable), 1),
        "false_answer_rate": sum(row["candidate_accepted"] for row in rows if not is_answerable(row)) / max(sum(not is_answerable(row) for row in rows), 1),
    }


def load_head_state(path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    # Training checkpoints retain argparse Paths as harmless metadata.
    with torch.serialization.safe_globals([PosixPath]):
        return torch.load(path, map_location=device, weights_only=True)["head"]


def load_verifier_state(path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    with torch.serialization.safe_globals([PosixPath]):
        return torch.load(path, map_location=device, weights_only=True)["verifier"]


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
    parser.add_argument("--nli", action="store_true", help="Use a support/refute/neutral verifier checkpoint.")
    parser.add_argument("--adapter-rank", type=int, default=0, help="Load a frozen-reader verification adapter from an NLI checkpoint.")
    parser.add_argument("--decoder-verifier", action="store_true", help="Use the adapter's decoder cross-attended verification token.")
    parser.add_argument("--evidence-radius", type=int, default=0, help="Include this many source tokens around the copied span as verifier evidence.")
    parser.add_argument("--claim", action="store_true", help="Render question/candidate as the declarative claim used by NLI training.")
    parser.add_argument("--joint-checkpoint", type=Path, help="Joint reader/verifier checkpoint; replaces the frozen reader with its shared reader state.")
    args = parser.parse_args()
    if args.joint_checkpoint and not args.nli:
        parser.error("--joint-checkpoint requires --nli")

    report = json.loads(args.input.read_text())
    rows = report["examples"]
    tokenizer = NeedleTokenizer(args.tokenizer, append_markers=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = NeedlePointerModel(NeedleConfig.public_checkpoint()).to(device=device, dtype=dtype)
    model.load_backbone_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=False)["model"])
    joint_state = torch.load(args.joint_checkpoint, map_location=device, weights_only=False) if args.joint_checkpoint else None
    if joint_state is not None:
        model.load_backbone_state_dict(joint_state["model"])
    encoder = NeedleVerifierAdapter(model, args.adapter_rank, decoder=args.decoder_verifier).to(device=device, dtype=dtype) if args.adapter_rank else model
    if args.adapter_rank:
        adapter_state = torch.load(args.head, map_location=device, weights_only=False)
        encoder.adapters.load_state_dict(adapter_state["adapter"])
        if args.decoder_verifier:
            encoder.decoder_adapters.load_state_dict(adapter_state["decoder_adapter"])
    encoder.eval()
    state = joint_state["verifier"] if joint_state is not None else load_verifier_state(args.head, device) if args.nli else load_head_state(args.head, device)
    hidden_dim = state["0.weight"].shape[0] if args.nli and not args.hidden_dim else args.hidden_dim
    head = (
        nn.Sequential(nn.Linear(NeedleConfig.public_checkpoint().d_model if args.decoder_verifier else NeedleConfig.public_checkpoint().d_model * 4, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 3))
        if args.nli
        else candidate_verifier_head(NeedleConfig.public_checkpoint().d_model, hidden_dim)
    ).to(device=device, dtype=dtype)
    head.load_state_dict(state)
    head.eval()

    eligible: list[tuple[int, list[int], int, int, int]] = []
    probabilities = [0.0] * len(rows)
    for index, row in enumerate(rows):
        candidate = row.get("raw_prediction", row.get("prediction", "")).strip()
        offset = candidate_offset(row["context"], candidate)
        if offset < 0:
            continue
        query = verifier_claim(row["question"], candidate) if args.claim else verifier_query(row["question"], candidate)
        window = _evidence_window(tokenizer, query, row["context"], offset, offset + len(candidate), max_source_length=SOURCE_LENGTH)
        if window is not None:
            candidate_positions = window[2]
            candidate_start = window[3] + candidate_positions[0]
            candidate_end = window[3] + candidate_positions[-1] + 1
            eligible.append((index, window[0], window[3], max(window[3], candidate_start - args.evidence_radius), min(len(window[0]), candidate_end + args.evidence_radius)))

    for start in range(0, len(eligible), args.batch_size):
        batch = eligible[start : start + args.batch_size]
        width = max(len(ids) for _, ids, _, _, _ in batch)
        source = torch.zeros((len(batch), width), dtype=torch.long, device=device)
        valid = torch.zeros_like(source, dtype=torch.bool)
        context = torch.zeros_like(source, dtype=torch.bool)
        candidate = torch.zeros_like(source, dtype=torch.bool)
        for row_index, (output_index, ids, context_start, candidate_start, candidate_end) in enumerate(batch):
            source[row_index, : len(ids)] = torch.tensor(ids, device=device)
            valid[row_index, : len(ids)] = True
            context[row_index, context_start : len(ids)] = True
            candidate[row_index, candidate_start:candidate_end] = True
        memory = encoder.encode(source, valid) if not args.decoder_verifier else None
        scores = (
            head(encoder.verify(source, valid, (valid & ~context) | candidate) if args.decoder_verifier else candidate_span_features(memory, valid, valid & ~context, candidate)).softmax(dim=-1)[:, 2]
            if args.nli
            else head(answerability_interaction_features(memory, valid, context)).squeeze(-1).sigmoid()
        ).float().cpu().tolist()
        for (output_index, _, _, _, _), score in zip(batch, scores):
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
    summary = safe_gate_summary(verified)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"input": str(args.input), "threshold": args.threshold, "summary": summary, "examples": verified}, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
