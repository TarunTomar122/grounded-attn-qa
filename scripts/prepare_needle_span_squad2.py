from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset

from grounded_qa.needle_qa_data import (
    SQUAD2_DATASET,
    SQUAD2_REVISION,
    prepare_squad2_item,
    prepare_squad2_unanswerable,
)
from grounded_qa.needle_tokenizer import NeedleTokenizer


SOURCE_LENGTH = 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensorize(rows: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    source_ids = torch.zeros((len(rows), SOURCE_LENGTH), dtype=torch.int16)
    source_lengths = torch.zeros(len(rows), dtype=torch.int16)
    context_start = torch.zeros(len(rows), dtype=torch.int16)
    gold_start = torch.zeros(len(rows), dtype=torch.int64)
    gold_end = torch.zeros(len(rows), dtype=torch.int64)
    answerable = torch.zeros(len(rows), dtype=torch.bool)
    for index, row in enumerate(rows):
        ids = row["source_ids"]
        source_ids[index, : len(ids)] = torch.tensor(ids, dtype=torch.int16)
        source_lengths[index] = len(ids)
        context_start[index] = row["context_start"]
        gold_start[index] = row["gold_start"]
        gold_end[index] = row["gold_end"]
        answerable[index] = row["answerable"]
    return {
        "source_ids": source_ids,
        "source_lengths": source_lengths,
        "context_start": context_start,
        "gold_start": gold_start,
        "gold_end": gold_end,
        "answerable": answerable,
    }


def prepare_split(dataset, tokenizer: NeedleTokenizer, *, limit: int | None = None) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    stats: dict[str, Any] = {
        "official_rows": len(dataset),
        "input_rows": 0,
        "kept_rows": 0,
        "kept_answerable": 0,
        "kept_unanswerable": 0,
        "dropped": {},
        "source_tokens": {"min": None, "max": 0, "total": 0},
    }
    for index, item in enumerate(dataset):
        if limit is not None and stats["input_rows"] >= limit:
            break
        stats["input_rows"] += 1
        is_answerable = bool(item.get("answers", {}).get("text", []))
        if is_answerable:
            example, reason = prepare_squad2_item(
                item,
                tokenizer,
                example_index=index,
                max_source_length=SOURCE_LENGTH,
            )
        else:
            example, reason = prepare_squad2_unanswerable(
                item,
                tokenizer,
                example_index=index,
                max_source_length=SOURCE_LENGTH,
            )
        if example is None:
            stats["dropped"][reason or "unknown"] = stats["dropped"].get(reason or "unknown", 0) + 1
            continue

        if is_answerable:
            copy_positions = [position for position in example.gold_copy_positions if position >= 0]
            if not copy_positions:
                stats["dropped"]["missing_copy_positions"] = stats["dropped"].get("missing_copy_positions", 0) + 1
                continue
            gold_start = copy_positions[0] + 1
            gold_end = copy_positions[-1] + 1
        else:
            gold_start = gold_end = 0

        source_length = len(example.source_ids)
        rows.append(
            {
                "source_ids": example.source_ids,
                "source_length": source_length,
                "context_start": example.context_start,
                "gold_start": gold_start,
                "gold_end": gold_end,
                "answerable": is_answerable,
            }
        )
        answers = item.get("answers", {})
        metadata.append(
            {
                "id": str(item.get("id", index)),
                "title": str(item.get("title", "")),
                "question": str(item.get("question", "")),
                "context": str(item.get("context", "")),
                "answers": {
                    "text": [str(value) for value in answers.get("text", [])],
                    "answer_start": [int(value) for value in answers.get("answer_start", [])],
                },
                "answerable": is_answerable,
                "source_length": source_length,
                "context_start": example.context_start,
                "gold_start": gold_start,
                "gold_end": gold_end,
                "window_start": example.window_start,
            }
        )
        stats["kept_rows"] += 1
        stats["kept_answerable" if is_answerable else "kept_unanswerable"] += 1
        lengths = stats["source_tokens"]
        lengths["min"] = source_length if lengths["min"] is None else min(lengths["min"], source_length)
        lengths["max"] = max(lengths["max"], source_length)
        lengths["total"] += source_length

    if stats["kept_rows"]:
        stats["source_tokens"]["mean"] = stats["source_tokens"]["total"] / stats["kept_rows"]
    else:
        stats["source_tokens"]["mean"] = 0.0
    return tensorize(rows), metadata, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the clean Needle SQuAD2 span-plus-NULL experiment.")
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None, help="Optional deterministic prefix limit for mechanical gates.")
    args = parser.parse_args()

    tokenizer = NeedleTokenizer(args.tokenizer, append_markers=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "purpose": "Clean joint extractive SQuAD2 training for pretrained Needle: answer span or first-class NULL.",
        "dataset": {"name": SQUAD2_DATASET, "revision": SQUAD2_REVISION},
        "tokenizer": {"path": str(args.tokenizer), "sha256": sha256(args.tokenizer), "append_markers": False},
        "format": {
            "source": "NeedleTokenizer.encode_source(query, context) = query_ids + [5] + context_ids",
            "source_length": SOURCE_LENGTH,
            "null_class": 0,
            "gold_positions": "1 + absolute source position; 0 means NULL",
            "answerable_window": "the deterministic tokenizer-aligned evidence window; answer spans are never truncated",
        },
        "splits": {},
    }
    for split in ("train", "validation"):
        dataset = load_dataset(SQUAD2_DATASET, split=split, revision=SQUAD2_REVISION)
        tensors, metadata, stats = prepare_split(dataset, tokenizer, limit=args.limit)
        torch.save(tensors, args.output_dir / f"{split}.pt")
        with (args.output_dir / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for row in metadata:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        manifest["splits"][split] = stats
        print(json.dumps({"split": split, **stats}, sort_keys=True), flush=True)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
