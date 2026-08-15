from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import torch
from torch import nn

from grounded_qa.metrics import exact_match, repeated_ngram_rate, token_f1, unsupported_entity_rate, unsupported_number_rate
from grounded_qa.negatives import REFUSAL
from grounded_qa.needle_pointer import (
    NeedleAnswerablePointerModel,
    NeedlePointerModel,
    NeedlePointerOutput,
    answerability_interaction_features,
)
from grounded_qa.needle_tokenizer import EOS_ID, NeedleTokenizer
from grounded_qa.needleish import NeedleConfig, NeedleishModel, load_public_checkpoint


def apply_refusal(prediction: str, probability: float, threshold: float) -> str:
    return prediction if probability >= threshold else REFUSAL


@torch.inference_mode()
def generate_batch(
    model: NeedleishModel | NeedlePointerModel,
    source_ids: torch.Tensor,
    source_valid: torch.Tensor,
    max_new_tokens: int,
    context_mask: torch.Tensor | None = None,
    memory: torch.Tensor | None = None,
) -> torch.Tensor:
    memory = model.encode(source_ids, source_valid) if memory is None else memory
    decoder = torch.full((source_ids.shape[0], 1), EOS_ID, dtype=torch.long, device=source_ids.device)
    finished = torch.zeros(source_ids.shape[0], dtype=torch.bool, device=source_ids.device)
    for _ in range(max_new_tokens):
        target_valid = torch.ones_like(decoder, dtype=torch.bool)
        if isinstance(model, NeedlePointerModel):
            if context_mask is None:
                raise ValueError("Pointer decoding requires a context mask")
            output = model.decode_pointer(decoder, memory, source_ids, source_valid, context_mask, target_valid)
            last = NeedlePointerOutput(
                output.vocab_logits[:, -1:],
                output.copy_position_probs[:, -1:],
                output.p_gen[:, -1:],
            )
            next_ids = last.final_distribution(source_ids)[:, 0].argmax(dim=-1)
        else:
            logits = model.decode(decoder, memory, source_valid, target_valid)
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
    if rows and all("answerable" in row and "refused" in row for row in rows):
        tp = sum(not row["refused"] and row["answerable"] for row in rows)
        tn = sum(row["refused"] and not row["answerable"] for row in rows)
        fp = sum(not row["refused"] and not row["answerable"] for row in rows)
        fn = sum(row["refused"] and row["answerable"] for row in rows)
        summary["answerability"] = {
            "answerability_f1": 2 * tp / max(2 * tp + fp + fn, 1),
            "refusal_precision": tn / max(tn + fn, 1),
            "refusal_recall": tn / max(tn + fp, 1),
            "refusal_f1": 2 * tn / max(2 * tn + fp + fn, 1),
            "false_refusal_rate": fn / max(tp + fn, 1),
            "hallucinated_answer_rate": fp / max(tn + fp, 1),
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
    parser.add_argument("--pointer", action="store_true")
    parser.add_argument("--answerability", action="store_true")
    parser.add_argument("--answerability-threshold", type=float, default=0.5)
    parser.add_argument("--interaction-head", type=Path, help="Optional frozen question/context interaction gate checkpoint.")
    args = parser.parse_args()
    if args.interaction_head and not args.answerability:
        parser.error("--interaction-head requires --answerability")

    input_bytes = args.input.read_bytes()
    rows = [json.loads(line) for line in input_bytes.decode().splitlines() if line]
    tokenizer = NeedleTokenizer(args.tokenizer, append_markers=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    pointer = args.pointer or args.answerability
    model_class = NeedleAnswerablePointerModel if args.answerability else NeedlePointerModel if pointer else NeedleishModel
    model = model_class(NeedleConfig.public_checkpoint()).to(device=device, dtype=dtype)
    if args.checkpoint.suffix == ".safetensors":
        if pointer:
            from safetensors.torch import load_file

            model.load_state_dict(load_file(str(args.checkpoint), device=str(device)))
        else:
            load_public_checkpoint(model, args.checkpoint)
    else:
        state = torch.load(args.checkpoint, map_location=device, weights_only=False)["model"]
        if pointer:
            model.load_backbone_state_dict(state)
        else:
            model.load_state_dict(state)
    model.eval()
    interaction_head = None
    if args.interaction_head:
        interaction_head = nn.Linear(NeedleConfig.public_checkpoint().d_model * 4, 1).to(device=device, dtype=dtype)
        interaction_head.load_state_dict(torch.load(args.interaction_head, map_location=device, weights_only=True)["head"])
        interaction_head.eval()

    results = []
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        encoded = [tokenizer.encode_source(row["question"], row["context"]) for row in batch]
        if any(len(ids) > 1024 for ids in encoded):
            raise ValueError("Evaluation source exceeds the public Needle 1024-token limit")
        width = max(map(len, encoded))
        source = torch.zeros((len(batch), width), dtype=torch.long, device=device)
        valid = torch.zeros_like(source, dtype=torch.bool)
        context_mask = torch.zeros_like(source, dtype=torch.bool)
        for index, ids in enumerate(encoded):
            source[index, : len(ids)] = torch.tensor(ids, device=device)
            valid[index, : len(ids)] = True
            context_start = len(tokenizer.encode(batch[index]["question"])) + 1
            context_mask[index, context_start : len(ids)] = True
        memory = model.encode(source, valid)
        probabilities = [1.0] * len(batch)
        if isinstance(model, NeedleAnswerablePointerModel):
            logits = (
                interaction_head(answerability_interaction_features(memory, valid, context_mask)).squeeze(-1)
                if interaction_head is not None
                else model.classify_answerability(model.evidence_position_logits(model.decode_pointer(
                    torch.full((len(batch), 1), EOS_ID, dtype=torch.long, device=device),
                    memory,
                    source,
                    valid,
                    context_mask,
                    torch.ones((len(batch), 1), dtype=torch.bool, device=device),
                ).copy_position_logits))
            )
            probabilities = logits.sigmoid().cpu().tolist()
        generated = generate_batch(model, source, valid, args.max_new_tokens, context_mask, memory).cpu().tolist()
        for row, ids, probability in zip(batch, generated, probabilities):
            eos = EOS_ID in ids
            if eos:
                ids = ids[: ids.index(EOS_ID)]
            raw_prediction = tokenizer.decode(ids).strip()
            prediction = apply_refusal(raw_prediction, probability, args.answerability_threshold) if args.answerability else raw_prediction
            em = max(exact_match(prediction, answer) for answer in row["answers"])
            f1 = max(token_f1(prediction, answer) for answer in row["answers"])
            result = {
                **row,
                "prediction": prediction,
                "generated_ids": ids,
                "generated_tokens": len(ids),
                "eos": eos,
                "em": em,
                "token_f1": f1,
                "unsupported_number_rate": unsupported_number_rate(prediction, row["context"]),
                "unsupported_entity_rate": unsupported_entity_rate(prediction, row["context"], row["question"]),
            }
            if args.answerability:
                result.update({
                    "raw_prediction": raw_prediction,
                    "p_answerable": probability,
                    "answerable": row.get("answerable", row["condition"] in {"correct", "counterfactual"}),
                    "refused": probability < args.answerability_threshold,
                })
            results.append(result)

    report = {
        "checkpoint": str(args.checkpoint),
        "evaluation_set": {"path": str(args.input), "sha256": hashlib.sha256(input_bytes).hexdigest(), "rows": len(rows)},
        "model": NeedleConfig.public_checkpoint().to_dict(),
        "decoding": {
            "method": "raw greedy with refusal gate" if args.answerability else "raw greedy",
            "pointer": pointer,
            "answerability": args.answerability,
            "answerability_gate": "interaction" if args.interaction_head else "model_head" if args.answerability else None,
            "answerability_threshold": args.answerability_threshold if args.answerability else None,
            "decoder_start_id": EOS_ID,
            "max_new_tokens": args.max_new_tokens,
        },
        "summary": summarize(results),
        "examples": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
