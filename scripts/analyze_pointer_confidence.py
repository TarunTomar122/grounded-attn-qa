from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from grounded_qa.calibration import choose_threshold, sweep_thresholds
from grounded_qa.needle_pointer import NeedleAnswerablePointerModel, NeedlePointerOutput
from grounded_qa.needle_tokenizer import EOS_ID, NeedleTokenizer
from grounded_qa.needleish import NeedleConfig


def binary_auc(scores: list[float], labels: list[bool]) -> float:
    positives = [score for score, label in zip(scores, labels) if label]
    negatives = [score for score, label in zip(scores, labels) if not label]
    if not positives or not negatives:
        raise ValueError("AUC requires both answerable and unanswerable rows")
    wins = sum(positive > negative for positive in positives for negative in negatives)
    ties = sum(positive == negative for positive in positives for negative in negatives)
    return (wins + 0.5 * ties) / (len(positives) * len(negatives))


def summarize_signal(scores: list[float], labels: list[bool]) -> dict[str, float]:
    safe = choose_threshold(sweep_thresholds(scores, labels), max_false_answer_rate=0.02)
    return {
        "auc": binary_auc(scores, labels),
        "safe_threshold": safe.threshold,
        "safe_answer_coverage": safe.answer_coverage,
        "safe_false_answer_rate": safe.false_answer_rate,
    }


@torch.inference_mode()
def score_batch(model, tokenizer, rows: list[dict], device: torch.device, max_new_tokens: int) -> list[dict]:
    encoded = [tokenizer.encode_source(row["question"], row["context"]) for row in rows]
    if any(len(ids) > 1024 for ids in encoded):
        raise ValueError("source exceeds the public Needle 1024-token limit")
    width = max(map(len, encoded))
    source = torch.zeros((len(rows), width), dtype=torch.long, device=device)
    valid = torch.zeros_like(source, dtype=torch.bool)
    context_mask = torch.zeros_like(source, dtype=torch.bool)
    for index, ids in enumerate(encoded):
        source[index, : len(ids)] = torch.tensor(ids, device=device)
        valid[index, : len(ids)] = True
        context_start = len(tokenizer.encode(rows[index]["question"])) + 1
        context_mask[index, context_start : len(ids)] = True

    memory = model.encode(source, valid)
    decoder = torch.full((len(rows), 1), EOS_ID, dtype=torch.long, device=device)
    selection = model.decode_pointer(decoder, memory, source, valid, context_mask, torch.ones_like(decoder, dtype=torch.bool))
    head = model.classify_answerability(model.evidence_position_logits(selection.copy_position_logits)).sigmoid()
    finished = torch.zeros(len(rows), dtype=torch.bool, device=device)
    count = torch.zeros(len(rows), dtype=torch.long, device=device)
    totals = {name: torch.zeros(len(rows), device=device) for name in ("final", "copy_token", "position_peak")}
    first = {name: torch.zeros(len(rows), device=device) for name in totals}

    for _ in range(max_new_tokens):
        output = model.decode_pointer(decoder, memory, source, valid, context_mask, torch.ones_like(decoder, dtype=torch.bool))
        last = NeedlePointerOutput(output.vocab_logits[:, -1:], output.copy_position_probs[:, -1:], output.p_gen[:, -1:])
        final = last.final_distribution(source)[:, 0]
        next_ids = final.argmax(dim=-1)
        active = ~finished
        emitted = active & next_ids.ne(EOS_ID)
        copy_probs = output.copy_position_probs[:, -1]
        copy_token = (copy_probs * source.eq(next_ids[:, None])).sum(dim=-1)
        values = {
            "final": final.gather(1, next_ids[:, None]).squeeze(1),
            "copy_token": copy_token,
            "position_peak": copy_probs.max(dim=-1).values,
        }
        is_first = emitted & count.eq(0)
        for name, value in values.items():
            totals[name] += value * emitted
            first[name] = torch.where(is_first, value, first[name])
        count += emitted.long()
        finished |= next_ids.eq(EOS_ID)
        decoder = torch.cat((decoder, torch.where(finished, torch.full_like(next_ids, EOS_ID), next_ids)[:, None]), dim=1)
        if finished.all():
            break

    result = []
    for index, row in enumerate(rows):
        item = {
            "id": row.get("id", str(index)),
            "condition": row["condition"],
            "answerable": row.get("answerable", row["condition"] in {"correct", "counterfactual"}),
            "answerability_probability": float(head[index]),
            "generated_tokens": int(count[index]),
        }
        for name, total in totals.items():
            item[f"first_{name}_probability"] = float(first[name][index])
            item[f"mean_{name}_probability"] = float(total[index] / count[index].clamp_min(1))
        result.append(item)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare inference-time grounding confidence signals.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    input_bytes = args.input.read_bytes()
    rows = [json.loads(line) for line in input_bytes.decode().splitlines() if line]
    tokenizer = NeedleTokenizer(args.tokenizer, append_markers=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeedleAnswerablePointerModel(NeedleConfig.public_checkpoint()).to(device=device, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=False)["model"])
    model.eval()
    scored = [
        item
        for start in range(0, len(rows), args.batch_size)
        for item in score_batch(model, tokenizer, rows[start : start + args.batch_size], device, args.max_new_tokens)
    ]
    labels = [item["answerable"] for item in scored]
    signals = [
        "answerability_probability",
        "first_final_probability",
        "mean_final_probability",
        "first_copy_token_probability",
        "mean_copy_token_probability",
        "first_position_peak_probability",
        "mean_position_peak_probability",
    ]
    report = {
        "checkpoint": str(args.checkpoint),
        "evaluation_set": {"path": str(args.input), "rows": len(rows), "sha256": hashlib.sha256(input_bytes).hexdigest()},
        "signals": {signal: summarize_signal([item[signal] for item in scored], labels) for signal in signals},
        "rows": scored,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["signals"], indent=2))


if __name__ == "__main__":
    main()
