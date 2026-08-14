from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

from grounded_qa.needle_tokenizer import NeedleTokenizer
from grounded_qa.synth_data import encode_synth_row, split_for_source


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--shards", nargs="+", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output-dir", default="artifacts/synth_day1")
    parser.add_argument("--source-max", type=int, default=512)
    parser.add_argument("--target-max", type=int, default=256)
    args = parser.parse_args()
    tokenizer = NeedleTokenizer(args.tokenizer)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    handles = {
        "train": (output_dir / "train.jsonl").open("w"),
        "validation": (output_dir / "validation.jsonl").open("w"),
    }
    counts: Counter[str] = Counter()
    exercise_counts: dict[str, Counter[str]] = {"train": Counter(), "validation": Counter()}
    source_urls: dict[str, set[str]] = {"train": set(), "validation": set()}
    token_totals: Counter[str] = Counter()
    try:
        for shard in args.shards:
            path = Path(args.input_dir) / shard
            parquet = pq.ParquetFile(path)
            counts[f"rows_{shard}"] = parquet.metadata.num_rows
            for batch in parquet.iter_batches(batch_size=4096):
                for row in batch.to_pylist():
                    if row.get("language") != "en":
                        counts["drop_non_english"] += 1
                        continue
                    if any(not isinstance(row.get(field), str) or not row[field].strip() for field in ("query", "query_seed_text", "synthetic_reasoning", "synthetic_answer")):
                        counts["drop_missing_required"] += 1
                        continue
                    encoded = encode_synth_row(row, tokenizer, source_max_length=args.source_max, target_max_length=args.target_max)
                    if encoded is None:
                        counts["drop_length"] += 1
                        continue
                    split = split_for_source(encoded.source_url)
                    handles[split].write(json.dumps(encoded.to_dict(), ensure_ascii=False) + "\n")
                    counts[f"kept_{split}"] += 1
                    exercise_counts[split][encoded.exercise] += 1
                    source_urls[split].add(encoded.source_url)
                    token_totals["source_tokens"] += encoded.source_tokens
                    token_totals["target_tokens"] += encoded.target_tokens
                    token_totals["reasoning_tokens"] += encoded.reasoning_tokens
                    token_totals["answer_tokens"] += encoded.answer_tokens
    finally:
        for handle in handles.values():
            handle.close()

    intersection = source_urls["train"] & source_urls["validation"]
    manifest = {
        "dataset": "PleIAs/SYNTH",
        "shards": args.shards,
        "language": "en",
        "source_max_length": args.source_max,
        "target_max_length": args.target_max,
        "counts": dict(counts),
        "exercise_distribution": {split: dict(values) for split, values in exercise_counts.items()},
        "unique_source_urls": {split: len(values) for split, values in source_urls.items()},
        "source_url_intersection": len(intersection),
        "token_totals": dict(token_totals),
        "source_disjoint": not intersection,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
