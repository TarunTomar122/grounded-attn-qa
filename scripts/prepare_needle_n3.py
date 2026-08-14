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


def main() -> None:
    parser = argparse.ArgumentParser(description="Add real SQuAD2 negatives to the matched N2 corpus for N3.")
    parser.add_argument("--n2-data-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    tokenizer = NeedleTokenizer(args.tokenizer, append_markers=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "purpose": "N3 supervised answerability: preserve all N2 positives and add every usable official SQuAD2 negative once.",
        "negative_dataset": {"name": SQUAD2_DATASET, "revision": SQUAD2_REVISION},
        "n2_manifest_sha256": sha256(args.n2_data_dir / "n2-manifest.json"),
        "splits": {},
    }
    for split in ("train", "validation"):
        positives = load_split(args.n2_data_dir, split)
        dataset = load_dataset(SQUAD2_DATASET, split=split, revision=SQUAD2_REVISION)
        negatives = []
        dropped = 0
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
        positives["answerable"] = torch.ones(len(positives["source_ids"]), dtype=torch.bool)
        negative_tensors["answerable"] = torch.zeros(len(negatives), dtype=torch.bool)
        combined = {key: torch.cat((positives[key], negative_tensors[key])) for key in positives}
        path = args.output_dir / f"n3-{split}.pt"
        torch.save(combined, path)
        manifest["splits"][split] = {
            "answerable_rows": len(positives["source_ids"]),
            "unanswerable_rows": len(negatives),
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
