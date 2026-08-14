from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from datasets import load_dataset

from grounded_qa.needle_qa_data import (
    COQA_DATASET,
    COQA_REVISION,
    SQUAD2_DATASET,
    SQUAD2_REVISION,
    PreparedNeedleQA,
    prepare_coqa_split,
    prepare_squad2_split,
)
from grounded_qa.needle_tokenizer import EOS_ID, NeedleTokenizer


SOURCE_LENGTH = 1024
TARGET_LENGTH = 512


def tensorize(examples: list[PreparedNeedleQA]) -> dict[str, torch.Tensor]:
    rows = len(examples)
    source_ids = torch.zeros((rows, SOURCE_LENGTH), dtype=torch.int16)
    target_ids = torch.zeros((rows, TARGET_LENGTH), dtype=torch.int16)
    gold_copy_positions = torch.full((rows, TARGET_LENGTH), -1, dtype=torch.int16)
    source_lengths = torch.empty(rows, dtype=torch.int16)
    target_lengths = torch.empty(rows, dtype=torch.int16)
    context_start = torch.empty(rows, dtype=torch.int16)
    example_index = torch.empty(rows, dtype=torch.int32)
    turn_index = torch.empty(rows, dtype=torch.int16)
    evidence_start = torch.empty(rows, dtype=torch.int32)
    evidence_end = torch.empty(rows, dtype=torch.int32)
    window_start = torch.empty(rows, dtype=torch.int32)

    for index, example in enumerate(examples):
        if len(example.source_ids) > SOURCE_LENGTH or len(example.target_ids) > TARGET_LENGTH:
            raise ValueError("example exceeds fixed tensor dimensions")
        if example.target_ids[-1] != EOS_ID or example.gold_copy_positions[-1] != -1:
            raise ValueError("every target must end in EOS with no copy supervision")
        source_ids[index, : len(example.source_ids)] = torch.tensor(example.source_ids, dtype=torch.int16)
        target_ids[index, : len(example.target_ids)] = torch.tensor(example.target_ids, dtype=torch.int16)
        gold_copy_positions[index, : len(example.gold_copy_positions)] = torch.tensor(
            example.gold_copy_positions, dtype=torch.int16
        )
        source_lengths[index] = len(example.source_ids)
        target_lengths[index] = len(example.target_ids)
        context_start[index] = example.context_start
        example_index[index] = example.example_index
        turn_index[index] = example.turn_index
        evidence_start[index] = example.evidence_start
        evidence_end[index] = example.evidence_end
        window_start[index] = example.window_start

    return {
        "source_ids": source_ids,
        "target_ids": target_ids,
        "source_lengths": source_lengths,
        "target_lengths": target_lengths,
        "context_start": context_start,
        "gold_copy_positions": gold_copy_positions,
        "example_index": example_index,
        "turn_index": turn_index,
        "evidence_start": evidence_start,
        "evidence_end": evidence_end,
        "window_start": window_start,
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare pinned SQuAD2 and CoQA tensors for public Needle N2.")
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    tokenizer = NeedleTokenizer(args.tokenizer, append_markers=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict = {
        "format": {
            "vocab_size": tokenizer.vocab_size,
            "source_length": SOURCE_LENGTH,
            "target_length": TARGET_LENGTH,
            "source": "NeedleTokenizer.encode_source(query, context) = query_ids + [5] + context_ids",
            "target": "natural answer token IDs followed by EOS=1",
            "context": "context_start through source_lengths; remaining cells are PAD=0",
            "gold_copy_positions": "absolute source positions; -1 means no defensible copy target",
            "coqa_alignment": (
                "Exact target token IDs are greedily matched left-to-right only within the annotated rationale; "
                "paraphrased, reordered, absent, and EOS tokens remain -1."
            ),
            "coqa_history_turns": "up to 4 most recent turns; oldest turns are dropped if evidence would not fit",
        },
        "tokenizer": {"path": str(args.tokenizer), "sha256": sha256(args.tokenizer)},
        "datasets": {},
    }

    jobs = (
        ("squad2", SQUAD2_DATASET, SQUAD2_REVISION, prepare_squad2_split),
        ("coqa", COQA_DATASET, COQA_REVISION, prepare_coqa_split),
    )
    for label, dataset_name, revision, prepare in jobs:
        dataset_report = {"name": dataset_name, "revision": revision, "splits": {}}
        manifest["datasets"][label] = dataset_report
        for split in ("train", "validation"):
            dataset = load_dataset(dataset_name, split=split, revision=revision)
            examples, stats = prepare(
                dataset,
                tokenizer,
                max_source_length=SOURCE_LENGTH,
                max_target_length=TARGET_LENGTH,
            )
            tensors = tensorize(examples)
            path = args.output_dir / f"n2-{label}-{split}.pt"
            torch.save(tensors, path)
            dataset_report["splits"][split] = {
                "official_rows": len(dataset),
                "dataset_fingerprint": getattr(dataset, "_fingerprint", None),
                "stats": stats,
                "file": {
                    "path": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                },
            }
            print(f"{label}/{split}: {stats['kept_rows']:,} rows -> {path}", flush=True)
            del dataset, examples, tensors

    manifest_path = args.output_dir / "n2-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(manifest_path)


if __name__ == "__main__":
    main()
