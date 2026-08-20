from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from grounded_qa.needle_qa_data import _covers_only_span_or_whitespace, immutable_pieces, span_piece_indices
from grounded_qa.needle_tokenizer import NeedleTokenizer


def load_raw_items(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(question["id"]): {**question, "context": paragraph["context"]}
        for article in payload["data"]
        for paragraph in article["paragraphs"]
        for question in paragraph["qas"]
    }


def map_plausible_span(
    item: dict[str, Any],
    tokenizer: NeedleTokenizer,
    *,
    context_start: int,
    context_length: int,
    window_start: int,
) -> tuple[int, int] | None:
    context = str(item.get("context", ""))
    answers = item.get("plausible_answers", [])
    if not answers:
        return None
    window_context = context[window_start:]
    pieces = immutable_pieces(tokenizer, window_context)
    window = pieces[:context_length]
    for answer in answers:
        text = str(answer.get("text", ""))
        start = int(answer.get("answer_start", -1))
        end = start + len(text)
        if start < 0 or not text or context[start:end] != text:
            continue
        window_start_offset = start - window_start
        window_end_offset = end - window_start
        positions = span_piece_indices(window, window_start_offset, window_end_offset)
        if not positions or not _covers_only_span_or_whitespace(window_context, window, positions, window_start_offset, window_end_offset):
            continue
        return context_start + positions[0] + 1, context_start + positions[-1] + 1
    return None


def augment_split(data_dir: Path, raw_path: Path, tokenizer: NeedleTokenizer) -> dict[str, int]:
    split = "train" if raw_path.name.startswith("train-") else "validation"
    rows_path = data_dir / f"{split}.pt"
    metadata_path = data_dir / f"{split}.jsonl"
    rows = torch.load(rows_path, map_location="cpu", weights_only=True)
    metadata = [json.loads(line) for line in metadata_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    raw_items = load_raw_items(raw_path)
    aux_start = torch.zeros(len(metadata), dtype=torch.int64)
    aux_end = torch.zeros(len(metadata), dtype=torch.int64)
    aux_available = torch.zeros(len(metadata), dtype=torch.bool)
    stats = {"rows": len(metadata), "answerable": 0, "plausible_available": 0, "plausible_unavailable": 0}

    for index, row in enumerate(metadata):
        item = raw_items.get(str(row["id"]))
        if item is None:
            raise KeyError(f"missing raw SQuAD2 item {row['id']}")
        if row["answerable"]:
            start, end = int(row["gold_start"]), int(row["gold_end"])
            stats["answerable"] += 1
        else:
            mapped = map_plausible_span(
                item,
                tokenizer,
                context_start=int(row["context_start"]),
                context_length=int(row["source_length"]) - int(row["context_start"]),
                window_start=int(row.get("window_start", 0)),
            )
            if mapped is None:
                stats["plausible_unavailable"] += 1
                continue
            start, end = mapped
            stats["plausible_available"] += 1
        aux_start[index] = start
        aux_end[index] = end
        aux_available[index] = True
        row["auxiliary_span_available"] = True
        row["auxiliary_start"] = start
        row["auxiliary_end"] = end

    rows["auxiliary_start"] = aux_start
    rows["auxiliary_end"] = aux_end
    rows["auxiliary_available"] = aux_available
    torch.save(rows, rows_path)
    metadata_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in metadata), encoding="utf-8")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Add tokenizer-aligned SQuAD2 plausible-answer targets to prepared rows.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--train-json", type=Path, required=True)
    parser.add_argument("--validation-json", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    args = parser.parse_args()
    tokenizer = NeedleTokenizer(args.tokenizer, append_markers=False)
    stats = {
        "train": augment_split(args.data_dir, args.train_json, tokenizer),
        "validation": augment_split(args.data_dir, args.validation_json, tokenizer),
    }
    print(json.dumps(stats, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
