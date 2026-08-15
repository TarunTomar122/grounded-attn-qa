from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset

from grounded_qa.needle_qa_data import SQUAD2_DATASET, SQUAD2_REVISION, prepare_squad2_unanswerable
from grounded_qa.needle_tokenizer import NeedleTokenizer
from scripts.prepare_needle_n2 import SOURCE_LENGTH, sha256, tensorize
from scripts.train_needle_n1 import load_split


def cross_pair_negatives(positives: dict[str, torch.Tensor], count: int, seed: int = 17) -> dict[str, torch.Tensor]:
    """Pair each retained question with a different retained context and mask copy targets."""
    rows, source_width = positives["source_ids"].shape
    if not 0 < count <= rows:
        raise ValueError("cross-pair count must be between one and the number of positives")
    generator = torch.Generator().manual_seed(seed)
    anchors = torch.randperm(rows, generator=generator)[:count]
    donors = torch.randperm(rows, generator=generator)[:count]
    donors = torch.where(donors.eq(anchors), (donors + 1) % rows, donors)
    negatives = {key: value.index_select(0, anchors).clone() for key, value in positives.items()}
    source_ids = torch.zeros_like(negatives["source_ids"])
    source_lengths = torch.empty_like(negatives["source_lengths"])
    for row, (anchor, donor) in enumerate(zip(anchors.tolist(), donors.tolist())):
        context_start = int(positives["context_start"][anchor])
        donor_start = int(positives["context_start"][donor])
        donor_length = int(positives["source_lengths"][donor])
        donor_tokens = min(donor_length - donor_start, source_width - context_start)
        source_ids[row, :context_start] = positives["source_ids"][anchor, :context_start]
        source_ids[row, context_start : context_start + donor_tokens] = positives["source_ids"][donor, donor_start : donor_start + donor_tokens]
        source_lengths[row] = context_start + donor_tokens
    negatives["source_ids"] = source_ids
    negatives["source_lengths"] = source_lengths
    negatives["target_ids"].zero_()
    negatives["target_ids"][:, 0] = 1
    negatives["target_lengths"].fill_(1)
    negatives["gold_copy_positions"].fill_(-1)
    for key in ("evidence_start", "evidence_end", "window_start"):
        if key in negatives:
            negatives[key].fill_(-1)
    if "answerable" in negatives:
        negatives["answerable"].fill_(False)
    return negatives


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare N3 answerability tensors from N2 positives and one negative source.")
    parser.add_argument("--n2-data-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--negative-mode", choices=("official", "cross-pair"), default="official")
    parser.add_argument("--cross-pair-ratio", type=float, default=3 / 7)
    args = parser.parse_args()
    if not 0 < args.cross_pair_ratio <= 1:
        parser.error("--cross-pair-ratio must be in (0, 1]")

    tokenizer = NeedleTokenizer(args.tokenizer, append_markers=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "purpose": "N3 supervised answerability: preserve N2 positives and add one deterministic negative source.",
        "negative_mode": args.negative_mode,
        "n2_manifest_sha256": sha256(args.n2_data_dir / "n2-manifest.json"),
        "splits": {},
    }
    if args.negative_mode == "official":
        manifest["negative_dataset"] = {"name": SQUAD2_DATASET, "revision": SQUAD2_REVISION}
    else:
        manifest["negative_dataset"] = {"source": "cross-pair", "ratio": args.cross_pair_ratio, "seed": 17}
    for split in ("train", "validation"):
        positives = load_split(args.n2_data_dir, split)
        positives["answerable"] = torch.ones(len(positives["source_ids"]), dtype=torch.bool)
        dropped = 0
        if args.negative_mode == "cross-pair":
            negative_tensors = cross_pair_negatives(
                positives,
                count=round(len(positives["source_ids"]) * args.cross_pair_ratio),
            )
        else:
            dataset = load_dataset(SQUAD2_DATASET, split=split, revision=SQUAD2_REVISION)
            negatives = []
            for index, item in enumerate(dataset):
                example, reason = prepare_squad2_unanswerable(
                    item,
                    tokenizer,
                    example_index=index,
                    max_source_length=SOURCE_LENGTH,
                )
                if example is not None:
                    negatives.append(example)
                elif reason != "answerable":
                    dropped += 1
            negative_tensors = tensorize(negatives)
            negative_tensors["answerable"] = torch.zeros(len(negatives), dtype=torch.bool)
        combined = {key: torch.cat((positives[key], negative_tensors[key])) for key in positives}
        path = args.output_dir / f"n3-{split}.pt"
        torch.save(combined, path)
        manifest["splits"][split] = {
            "answerable_rows": len(positives["source_ids"]),
            "unanswerable_rows": len(negative_tensors["source_ids"]),
            "dropped_unanswerable_rows": dropped,
            "total_rows": len(combined["source_ids"]),
            "file": {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)},
        }
        print(json.dumps({"split": split, **manifest["splits"][split]}), flush=True)

    path = args.output_dir / "n3-manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(path)


if __name__ == "__main__":
    main()
