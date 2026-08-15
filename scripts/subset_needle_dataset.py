"""Create a deterministic answerability-only slice of prepared Needle tensors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from scripts.train_needle_n1 import load_split


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def select_rows(
    data: dict[str, torch.Tensor],
    count: int,
    answerable: bool,
    seed: int,
) -> dict[str, torch.Tensor]:
    """Select exactly count rows with the requested answerability label."""
    if count < 1:
        raise ValueError("count must be positive")
    labels = data.get("answerable")
    if labels is None:
        raise ValueError("input tensors must contain an answerable field")
    if labels.ndim != 1:
        raise ValueError("answerable must be a one-dimensional tensor")

    rows = len(labels)
    if not all(isinstance(value, torch.Tensor) and value.ndim > 0 and len(value) == rows for value in data.values()):
        raise ValueError("every tensor field must have the same row dimension")

    candidates = torch.nonzero(labels.bool().eq(answerable), as_tuple=False).flatten()
    if len(candidates) < count:
        available = int(len(candidates))
        label = "answerable" if answerable else "unanswerable"
        raise ValueError(f"requested {count} {label} rows, but only {available} are available")

    generator = torch.Generator().manual_seed(seed)
    indices = candidates.index_select(0, torch.randperm(len(candidates), generator=generator)[:count])
    return {key: value.index_select(0, indices) for key, value in data.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--answerable", type=parse_bool, required=True, metavar="true|false")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    if input_dir == output_dir:
        parser.error("--output-dir must differ from --input-dir")

    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"input_dir": str(input_dir), "output_dir": str(output_dir), "count": args.count, "answerable": args.answerable, "splits": {}}
    for split_index, split in enumerate(("train", "validation")):
        data = load_split(input_dir, split)
        selected = select_rows(data, args.count, args.answerable, args.seed + split_index)
        path = output_dir / f"mechanics-{split}.pt"
        torch.save(selected, path)
        summary["splits"][split] = {"rows": len(selected["source_ids"]), "path": str(path)}

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
