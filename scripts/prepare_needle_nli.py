"""Prepare a bounded, reproducible SNLI stage for the verifier adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from datasets import load_dataset

from grounded_qa.needle_tokenizer import NeedleTokenizer
from scripts.prepare_needle_n2 import SOURCE_LENGTH, sha256
from scripts.prepare_needle_n3_verifier import nli_claim_query, tensorize


LABELS = {"neutral": 0, "contradiction": 1, "entailment": 2}
SNLI_LABELS = {0: 2, 1: 0, 2: 1}  # dataset IDs: entailment, neutral, contradiction


def nli_query(claim: str) -> str:
    return nli_claim_query(claim)


def label_id(label: int | str) -> int | None:
    if isinstance(label, str):
        return LABELS.get(label)
    return SNLI_LABELS.get(label)


def split_rows(rows, tokenizer: NeedleTokenizer, limit: int) -> tuple[list[tuple[list[int], int, bool]], dict[str, int]]:
    selected = []
    stats = {"seen": 0, "kept": 0, "too_long": 0, "invalid_label": 0}
    for row in rows:
        label = label_id(row["label"])
        if label is None:
            stats["invalid_label"] += 1
            continue
        ids = tokenizer.encode_source(nli_query(row["hypothesis"]), row["premise"])
        stats["seen"] += 1
        if len(ids) > SOURCE_LENGTH:
            stats["too_long"] += 1
            continue
        # tensorize's boolean label is deliberately repurposed below; keep its
        # stable source-layout implementation rather than duplicating it.
        selected.append((ids, len(tokenizer.encode(nli_query(row["hypothesis"]))) + 1, label == 2))
        if len(selected) == limit:
            break
    stats["kept"] = len(selected)
    return selected, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare deterministic balanced SNLI verifier tensors.")
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-rows", type=int, default=120_000)
    parser.add_argument("--validation-rows", type=int, default=12_000)
    args = parser.parse_args()

    tokenizer = NeedleTokenizer(args.tokenizer, append_markers=False)
    dataset = load_dataset("stanfordnlp/snli")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "purpose": "General NLI initialization for the frozen-reader verifier adapter.",
        "dataset": {"name": "stanfordnlp/snli", "license": "CC-BY-SA-4.0", "labels": LABELS},
        "format": "query is a claim; context is the SNLI premise; labels are neutral=0, refute=1, support=2.",
        "splits": {},
    }
    for split, limit in (("train", args.train_rows), ("validation", args.validation_rows)):
        rows, stats = split_rows(dataset[split], tokenizer, limit)
        tensors = tensorize(rows)
        labels = []
        for row in dataset[split]:
            label = label_id(row["label"])
            if label is not None and len(tokenizer.encode_source(nli_query(row["hypothesis"]), row["premise"])) <= SOURCE_LENGTH:
                labels.append(label)
                if len(labels) == len(rows):
                    break
        tensors["nli_label"] = torch.tensor(labels, dtype=torch.int64)
        tensors["evidence_start"] = tensors["context_start"].clone()
        tensors["evidence_end"] = tensors["source_lengths"].clone()
        path = args.output_dir / f"nli-snli-{split}.pt"
        torch.save(tensors, path)
        manifest["splits"][split] = {"stats": stats, "class_counts": torch.bincount(tensors["nli_label"], minlength=3).tolist(), "file": {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}}
    (args.output_dir / "nli-snli-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
