from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from datasets import load_dataset

from grounded_qa.needle_tokenizer import NeedleTokenizer


FIELDS = ["query", "query_seed_text", "synthetic_reasoning", "synthetic_answer", "query_seed_url", "language", "exercise"]


def percentiles(values: list[int]) -> dict[str, float | None]:
    if not values:
        return {key: None for key in ("p50", "p90", "p95", "p99")}
    values = sorted(values)
    return {
        "p50": float(statistics.quantiles(values, n=100, method="inclusive")[49]) if len(values) > 1 else float(values[0]),
        "p90": float(statistics.quantiles(values, n=100, method="inclusive")[89]) if len(values) > 1 else float(values[0]),
        "p95": float(statistics.quantiles(values, n=100, method="inclusive")[94]) if len(values) > 1 else float(values[0]),
        "p99": float(statistics.quantiles(values, n=100, method="inclusive")[98]) if len(values) > 1 else float(values[0]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--rows", type=int, default=50_000)
    parser.add_argument("--output", default="artifacts/synth_audit.json")
    args = parser.parse_args()
    tokenizer = NeedleTokenizer(args.tokenizer)
    dataset = load_dataset("PleIAs/SYNTH", split="train", streaming=True)
    language_counts: Counter[str] = Counter()
    exercise_counts: Counter[str] = Counter()
    missing_counts: Counter[str] = Counter()
    lengths: dict[str, list[int]] = defaultdict(list)
    examples: dict[str, list[dict[str, object]]] = defaultdict(list)
    source_urls: set[str] = set()
    english_rows = 0

    for index, row in enumerate(dataset):
        if index >= args.rows:
            break
        language = str(row.get("language") or "missing")
        exercise = str(row.get("exercise") or "missing")
        language_counts[language] += 1
        exercise_counts[exercise] += 1
        for field in FIELDS:
            value = row.get(field)
            if value is None or (isinstance(value, str) and not value.strip()):
                missing_counts[field] += 1
        source_url = row.get("query_seed_url")
        if isinstance(source_url, str) and source_url:
            source_urls.add(source_url)
        if len(examples[exercise]) < 3:
            examples[exercise].append({field: row.get(field) for field in FIELDS + ["constraints", "script"]})
        if language != "en":
            continue
        english_rows += 1
        for name, field in {
            "query": "query",
            "context": "query_seed_text",
            "reasoning": "synthetic_reasoning",
            "answer": "synthetic_answer",
        }.items():
            value = row.get(field)
            if isinstance(value, str) and value:
                lengths[name].append(len(tokenizer.encode(value)))

    report = {
        "dataset": "PleIAs/SYNTH",
        "rows_requested": args.rows,
        "rows_observed": sum(language_counts.values()),
        "english_rows": english_rows,
        "language_distribution": dict(language_counts),
        "exercise_distribution": dict(exercise_counts),
        "missing_field_rates": {
            field: missing_counts[field] / max(sum(language_counts.values()), 1) for field in FIELDS
        },
        "token_length_percentiles_english": {name: percentiles(values) for name, values in lengths.items()},
        "unique_query_seed_urls": len(source_urls),
        "examples_by_exercise": dict(examples),
        "tokenizer": str(Path(args.tokenizer)),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
