from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from grounded_qa.needle_full_span_qa import NeedleFullSpanNullModel
from grounded_qa.needle_span_qa import best_spans
from grounded_qa.needle_tokenizer import NeedleTokenizer
from grounded_qa.needleish import NeedleConfig, load_public_checkpoint
from scripts.evaluate_needle_span_null import score, summarize
from scripts.train_needle_span_null import answer_text, batch, load_metadata, load_rows


def load_model(path: Path, device: torch.device) -> NeedleFullSpanNullModel:
    model = NeedleFullSpanNullModel(NeedleConfig.public_checkpoint()).to(device)
    if path.suffix == ".safetensors":
        load_public_checkpoint(model.backbone, path)
    else:
        state = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"], strict=False)
    return model.eval()


@torch.inference_mode()
def collect(
    model: NeedleFullSpanNullModel,
    rows: dict[str, torch.Tensor],
    metadata: list[dict[str, Any]],
    tokenizer: NeedleTokenizer,
    device: torch.device,
    batch_size: int,
    precision: torch.dtype | None,
) -> dict[str, Any]:
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
        predicted_start = output.start_logits.argmax(dim=1)
        predicted_end = output.end_logits.argmax(dim=1)
        has = answerable.bool()
        no = ~has
        has_rows += int(has.sum())
        no_rows += int(no.sum())
        start_correct += int((predicted_start[has] == gold_start[has]).sum())
        end_correct += int((predicted_end[has] == gold_end[has]).sum())
        null_start += int((predicted_start[no] == 0).sum())
        null_end += int((predicted_end[no] == 0).sum())
        for local, start, end, row_margin in zip(
            range(len(indices)), best_start.tolist(), best_end.tolist(), margin.tolist()
        ):
            raw_predictions.append(answer_text(tokenizer, source[local].cpu(), start, end))
            margins.append(float(row_margin))

    thresholds = sorted(set(margins))
    candidates = thresholds[:: max(1, len(thresholds) // 200)] if len(thresholds) > 201 else thresholds
    sweep = [summarize(metadata, raw_predictions, margins, threshold) for threshold in candidates]
    best = max(sweep, key=lambda item: item["all_f1"])
    final_predictions = [
        "" if margin >= best["threshold"] else prediction
        for margin, prediction in zip(margins, raw_predictions)
    ]
    examples: list[dict[str, Any]] = []
    wanted = (
        next((i for i, row in enumerate(metadata) if row["answerable"] and score(final_predictions[i], row)[0]), None),
        next((i for i, row in enumerate(metadata) if not row["answerable"] and not final_predictions[i]), None),
        next((i for i, row in enumerate(metadata) if row["answerable"] and not score(final_predictions[i], row)[0]), None),
        next((i for i, row in enumerate(metadata) if not row["answerable"] and final_predictions[i]), None),
    )
    for label, index in zip(("correct_answer", "correct_null", "wrong_answer", "false_answer"), wanted):
        if index is not None:
            examples.append(
                {
                    "kind": label,
                    "question": metadata[index]["question"],
                    "context": metadata[index]["context"],
                    "gold": metadata[index]["answers"]["text"],
                    "prediction": final_predictions[index],
                    "raw_prediction": raw_predictions[index],
                    "null_margin": margins[index],
                }
            )
    return {
        "argmax": {
            "has_answer_start_accuracy": start_correct / max(has_rows, 1),
            "has_answer_end_accuracy": end_correct / max(has_rows, 1),
            "null_start_accuracy": null_start / max(no_rows, 1),
            "null_end_accuracy": null_end / max(no_rows, 1),
        },
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
    parser = argparse.ArgumentParser(description="Evaluate the full Needle decoder span+NULL checkpoint on SQuAD2.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    precision = torch.bfloat16 if device.type == "cuda" else None
    rows = load_rows(args.data_dir / "validation.pt")
    metadata = load_metadata(args.data_dir / "validation.jsonl")
    tokenizer = NeedleTokenizer(args.tokenizer, append_markers=False)
    result = collect(load_model(args.checkpoint, device), rows, metadata, tokenizer, device, args.batch_size, precision)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "examples"}, sort_keys=True))


if __name__ == "__main__":
    main()
