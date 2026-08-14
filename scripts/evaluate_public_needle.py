from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch

from grounded_qa.metrics import exact_match, repeated_ngram_rate, token_f1, unsupported_entity_rate, unsupported_number_rate
from grounded_qa.needle_tokenizer import EOS_ID, NeedleTokenizer
from grounded_qa.needleish import NeedleConfig, NeedleishModel, load_public_checkpoint


@torch.inference_mode()
def generate_batch(
    model: NeedleishModel,
    source_ids: torch.Tensor,
    source_valid: torch.Tensor,
    max_new_tokens: int,
) -> torch.Tensor:
    memory = model.encode(source_ids, source_valid)
    decoder = torch.full((source_ids.shape[0], 1), EOS_ID, dtype=torch.long, device=source_ids.device)
    finished = torch.zeros(source_ids.shape[0], dtype=torch.bool, device=source_ids.device)
    for _ in range(max_new_tokens):
        logits = model.decode(decoder, memory, source_valid, torch.ones_like(decoder, dtype=torch.bool))
        next_ids = logits[:, -1].argmax(dim=-1)
        next_ids = torch.where(finished, torch.full_like(next_ids, EOS_ID), next_ids)
        decoder = torch.cat((decoder, next_ids[:, None]), dim=1)
        finished |= next_ids.eq(EOS_ID)
        if finished.all():
            break
    return decoder[:, 1:]


def summarize(rows: list[dict]) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["condition"]].append(row)
    summary = {}
    for condition, members in sorted(groups.items()):
        summary[condition] = {
            "rows": len(members),
            "em": sum(row["em"] for row in members) / len(members),
            "token_f1": sum(row["token_f1"] for row in members) / len(members),
            "eos_rate": sum(row["eos"] for row in members) / len(members),
            "mean_tokens": sum(row["generated_tokens"] for row in members) / len(members),
            "repeated_3gram_rate": repeated_ngram_rate((row["prediction"] for row in members)),
            "unsupported_number_rate": sum(row["unsupported_number_rate"] for row in members) / len(members),
            "unsupported_entity_rate": sum(row["unsupported_entity_rate"] for row in members) / len(members),
        }
    paired = defaultdict(dict)
    for row in rows:
        paired[row["pair_id"]][row["condition"]] = row["prediction"]
    comparisons = [
        values
        for values in paired.values()
        if {"correct", "wrong", "empty"}.issubset(values)
    ]
    summary["context_dependency"] = {
        "pairs": len(comparisons),
        "correct_vs_wrong_output_change_rate": sum(v["correct"] != v["wrong"] for v in comparisons) / max(len(comparisons), 1),
        "correct_vs_empty_output_change_rate": sum(v["correct"] != v["empty"] for v in comparisons) / max(len(comparisons), 1),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Free-running evaluation of the public Needle checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line]
    tokenizer = NeedleTokenizer(args.tokenizer, append_markers=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = NeedleishModel(NeedleConfig.public_checkpoint()).to(device=device, dtype=dtype)
    if args.checkpoint.suffix == ".safetensors":
        load_public_checkpoint(model, args.checkpoint)
    else:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=False)["model"])
    model.eval()

    results = []
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        encoded = [tokenizer.encode_source(row["question"], row["context"]) for row in batch]
        if any(len(ids) > 1024 for ids in encoded):
            raise ValueError("Evaluation source exceeds the public Needle 1024-token limit")
        width = max(map(len, encoded))
        source = torch.zeros((len(batch), width), dtype=torch.long, device=device)
        valid = torch.zeros_like(source, dtype=torch.bool)
        for index, ids in enumerate(encoded):
            source[index, : len(ids)] = torch.tensor(ids, device=device)
            valid[index, : len(ids)] = True
        generated = generate_batch(model, source, valid, args.max_new_tokens).cpu().tolist()
        for row, ids in zip(batch, generated):
            eos = EOS_ID in ids
            if eos:
                ids = ids[: ids.index(EOS_ID)]
            prediction = tokenizer.decode(ids).strip()
            em = max(exact_match(prediction, answer) for answer in row["answers"])
            f1 = max(token_f1(prediction, answer) for answer in row["answers"])
            results.append({
                **row,
                "prediction": prediction,
                "generated_ids": ids,
                "generated_tokens": len(ids),
                "eos": eos,
                "em": em,
                "token_f1": f1,
                "unsupported_number_rate": unsupported_number_rate(prediction, row["context"]),
                "unsupported_entity_rate": unsupported_entity_rate(prediction, row["context"], row["question"]),
            })

    report = {
        "checkpoint": str(args.checkpoint),
        "model": NeedleConfig.public_checkpoint().to_dict(),
        "decoding": {"method": "raw greedy", "decoder_start_id": EOS_ID, "max_new_tokens": args.max_new_tokens},
        "summary": summarize(results),
        "examples": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
