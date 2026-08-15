from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import torch
from datasets import load_dataset

from grounded_qa.needle_qa_data import SQUAD2_DATASET, SQUAD2_REVISION, PreparedNeedleQA, prepare_squad2_item, prepare_squad2_unanswerable
from grounded_qa.needle_tokenizer import NeedleTokenizer
from scripts.prepare_needle_n2 import SOURCE_LENGTH, sha256, tensorize


def split_for_context(context: str) -> str:
    return "validation" if int(hashlib.sha256(context.encode()).hexdigest()[:8], 16) % 20 == 0 else "train"


def matched_squad2_pairs(items, tokenizer) -> tuple[dict[str, list[tuple[PreparedNeedleQA, bool]]], dict[str, int]]:
    """Keep only paragraphs containing both an answerable and an unanswerable question."""
    grouped: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for index, item in enumerate(items):
        grouped[str(item.get("context", ""))].append((index, item))
    rows = {"train": [], "validation": []}
    stats = {"paragraphs": len(grouped), "matched_paragraphs": 0, "dropped_pairs": 0}
    for context, group in grouped.items():
        positive = negative = None
        for index, item in group:
            if item.get("answers", {}).get("text", []):
                candidate, _ = prepare_squad2_item(item, tokenizer, example_index=index, max_source_length=SOURCE_LENGTH)
                if candidate is not None and positive is None:
                    positive = candidate
            else:
                candidate, _ = prepare_squad2_unanswerable(item, tokenizer, example_index=index, max_source_length=SOURCE_LENGTH)
                if candidate is not None and negative is None:
                    negative = candidate
        if positive is None or negative is None:
            stats["dropped_pairs"] += 1
            continue
        split = split_for_context(context)
        rows[split].extend(((positive, True), (negative, False)))
        stats["matched_paragraphs"] += 1
    return rows, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare paragraph-matched SQuAD2 answerability rows from source train data.")
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    tokenizer = NeedleTokenizer(args.tokenizer, append_markers=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prepared, stats = matched_squad2_pairs(
        load_dataset(SQUAD2_DATASET, split="train", revision=SQUAD2_REVISION), tokenizer
    )
    manifest = {
        "purpose": "N3 natural answerability control: answerable and unanswerable questions share each paragraph.",
        "dataset": {"name": SQUAD2_DATASET, "revision": SQUAD2_REVISION, "split": "train"},
        "split": "sha256(context) modulo 20; source validation is untouched",
        "stats": stats,
        "splits": {},
    }
    for split, rows in prepared.items():
        examples = [example for example, _ in rows]
        tensors = tensorize(examples)
        tensors["answerable"] = torch.tensor([label for _, label in rows], dtype=torch.bool)
        path = args.output_dir / f"n3-{split}.pt"
        torch.save(tensors, path)
        manifest["splits"][split] = {
            "rows": len(rows),
            "answerable_rows": int(tensors["answerable"].sum()),
            "unanswerable_rows": int((~tensors["answerable"]).sum()),
            "file": {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)},
        }
    path = args.output_dir / "n3-matched-manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(path)


if __name__ == "__main__":
    main()
