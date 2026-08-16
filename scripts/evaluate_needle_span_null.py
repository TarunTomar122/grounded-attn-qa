from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from grounded_qa.needle_span_qa import NeedleSpanNullModel, best_spans
from grounded_qa.needleish import NeedleConfig, load_public_checkpoint
from grounded_qa.needle_tokenizer import NeedleTokenizer
from scripts.train_needle_span_null import answer_text, batch, load_metadata, load_rows, normalize_answer, token_f1


def score(prediction: str, row: dict[str, Any]) -> tuple[float, float]:
    truths = row["answers"]["text"] or [""]
    return (
        max(float(normalize_answer(prediction) == normalize_answer(truth)) for truth in truths),
        max(token_f1(prediction, truth) for truth in truths),
    )


def summarize(
    rows: list[dict[str, Any]],
    raw_predictions: list[str],
    margins: list[float],
    threshold: float,
) -> dict[str, Any]:
    predictions = ["" if margin >= threshold else prediction for margin, prediction in zip(margins, raw_predictions)]
    groups = {
        "all": list(range(len(rows))),
        "has_answer": [i for i, row in enumerate(rows) if row["answerable"]],
        "no_answer": [i for i, row in enumerate(rows) if not row["answerable"]],
    }
    result: dict[str, Any] = {"threshold": threshold}
    for name, indices in groups.items():
        if not indices:
            continue
        em = sum(score(predictions[i], rows[i])[0] for i in indices) / len(indices)
        f1 = sum(score(predictions[i], rows[i])[1] for i in indices) / len(indices)
        result[f"{name}_em"] = em
        result[f"{name}_f1"] = f1
    has = groups["has_answer"]
    no = groups["no_answer"]
    result["no_answer_accuracy"] = sum(not predictions[i] for i in no) / max(len(no), 1)
    result["false_refusal_rate"] = sum(not predictions[i] for i in has) / max(len(has), 1)
    result["false_answer_rate"] = sum(bool(predictions[i]) for i in no) / max(len(no), 1)
    return result


def load_model(path: Path, device: torch.device) -> NeedleSpanNullModel:
    model = NeedleSpanNullModel(NeedleConfig.public_checkpoint()).to(device)
    if path.suffix == ".safetensors":
        load_public_checkpoint(model.backbone, path)
    else:
        state = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
    return model.eval()


@torch.inference_mode()
def collect(model, rows, metadata, tokenizer, device, batch_size, precision):
    raw_predictions: list[str] = []
    margins: list[float] = []
    start_correct = end_correct = null_start = null_end = 0
    has_rows = no_rows = 0
    for offset in range(0, len(rows["source_ids"]), batch_size):
        indices = torch.arange(offset, min(offset + batch_size, len(rows["source_ids"])))
        source, valid, context, gold_start, gold_end, answerable = batch(rows, indices, device)
        with torch.autocast(device_type="cuda", dtype=precision, enabled=precision is not None and device.type == "cuda"):
            output = model(source, valid, context)
        best_start, best_end, _, margin = best_spans(output)
        argmax_start = output.start_logits.argmax(dim=1)
        argmax_end = output.end_logits.argmax(dim=1)
        has_mask = answerable.bool()
        no_mask = ~has_mask
        has_rows += int(has_mask.sum())
        no_rows += int(no_mask.sum())
        start_correct += int((argmax_start[has_mask] == gold_start[has_mask]).sum())
        end_correct += int((argmax_end[has_mask] == gold_end[has_mask]).sum())
        null_start += int((argmax_start[no_mask] == 0).sum())
        null_end += int((argmax_end[no_mask] == 0).sum())
        for local, start, end, row_margin in zip(range(len(indices)), best_start.tolist(), best_end.tolist(), margin.tolist()):
            raw_predictions.append(answer_text(tokenizer, source[local].cpu(), start, end))
            margins.append(float(row_margin))
    thresholds = sorted(set(margins))
    candidates = thresholds[:: max(1, len(thresholds) // 200)] if len(thresholds) > 201 else thresholds
    sweep = [summarize(metadata, raw_predictions, margins, threshold) for threshold in candidates]
    best = max(sweep, key=lambda item: item["all_f1"])
    final_predictions = ["" if margin >= best["threshold"] else prediction for margin, prediction in zip(margins, raw_predictions)]
    examples: list[dict[str, Any]] = []
    wanted = (
        next((i for i, row in enumerate(metadata) if row["answerable"] and final_predictions[i] and score(final_predictions[i], row)[0]), None),
        next((i for i, row in enumerate(metadata) if not row["answerable"] and not final_predictions[i]), None),
        next((i for i, row in enumerate(metadata) if row["answerable"] and score(final_predictions[i], row)[0] == 0), None),
        next((i for i, row in enumerate(metadata) if not row["answerable"] and final_predictions[i]), None),
    )
    labels = ("correct_answer", "correct_null", "wrong_answer", "false_answer")
    for label, index in zip(labels, wanted):
        if index is not None:
            examples.append({"kind": label, "question": metadata[index]["question"], "context": metadata[index]["context"], "gold": metadata[index]["answers"]["text"], "prediction": final_predictions[index], "raw_prediction": raw_predictions[index], "null_margin": margins[index]})
    return {
        "argmax": {"has_answer_start_accuracy": start_correct / max(has_rows, 1), "has_answer_end_accuracy": end_correct / max(has_rows, 1), "null_start_accuracy": null_start / max(no_rows, 1), "null_end_accuracy": null_end / max(no_rows, 1)},
        "raw_best_span": summarize(metadata, raw_predictions, [0.0] * len(margins), float("inf")),
        "best_threshold": best,
        "threshold_sweep_top": sorted(sweep, key=lambda item: item["all_f1"], reverse=True)[:10],
        "margin_distribution": {
            "has_answer_mean": sum(margins[i] for i, row in enumerate(metadata) if row["answerable"]) / max(has_rows, 1),
            "no_answer_mean": sum(margins[i] for i, row in enumerate(metadata) if not row["answerable"]) / max(no_rows, 1),
        },
        "examples": examples,
        "rows": len(metadata),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a Needle span/null checkpoint on SQuAD2.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    precision = torch.bfloat16 if device.type == "cuda" else None
    model = load_model(args.checkpoint, device)
    rows = load_rows(args.data_dir / "validation.pt")
    metadata = load_metadata(args.data_dir / "validation.jsonl")
    tokenizer = NeedleTokenizer(args.tokenizer, append_markers=False)
    result = collect(model, rows, metadata, tokenizer, device, args.batch_size, precision)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "examples"}, sort_keys=True))


if __name__ == "__main__":
    main()
