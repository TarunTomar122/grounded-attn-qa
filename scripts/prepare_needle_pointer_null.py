"""Keep only examples whose answer/refusal can be supervised as a source choice."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from scripts.train_needle_n1 import load_split


def filter_pointer_null_rows(data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Retain negatives and answerable rows with an exact first source position."""
    if "answerable" not in data:
        raise ValueError("pointer-NULL data requires answerability labels")
    keep = ~data["answerable"] | data["gold_copy_positions"][:, 0].ge(0)
    return {key: value[keep] for key, value in data.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare exact-span answer-or-NULL tensors.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"source": str(args.input_dir), "rule": "all negatives plus positives with a gold first copy position", "splits": {}}
    for split in ("train", "validation"):
        data = load_split(args.input_dir, split)
        kept = filter_pointer_null_rows(data)
        path = args.output_dir / f"n3-{split}.pt"
        torch.save(kept, path)
        manifest["splits"][split] = {
            "input_rows": len(data["source_ids"]),
            "output_rows": len(kept["source_ids"]),
            "answerable_rows": int(kept["answerable"].sum()),
            "unanswerable_rows": int((~kept["answerable"]).sum()),
        }
    (args.output_dir / "pointer-null-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
