#!/usr/bin/env python3
"""Materialize only fixed evaluation data; Phase A1 training remains procedural and online."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grounded_qa.synthetic import diversity_stats, phase_a_test_set, phase_a_validation_splits


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="artifacts/phase-a1-v6")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-n", type=int, default=1_000)
    parser.add_argument("--test-n", type=int, default=10_000)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    splits = phase_a_validation_splits(args.val_n, args.seed)
    for name, rows in splits.items():
        write_jsonl(out_dir / f"{name}.jsonl", rows)
    test_rows = phase_a_test_set(args.test_n, args.seed)
    write_jsonl(out_dir / "test.jsonl", test_rows)
    manifest = {
        "dataset_version": "synthetic_procedural_copy_v6",
        "seed": args.seed,
        "training": {
            "mode": "online",
            "row_function": "procedural_copy_row(seed, global_example_index)",
            "materialized_rows": 0,
            "relations": ["identifier", "access_code", "serial_number", "registry_key"],
        },
        "validation": {name: {"rows": len(rows), **diversity_stats(rows)} for name, rows in splits.items()},
        "test": {"rows": len(test_rows), **diversity_stats(test_rows)},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
