from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

import duckdb


REPO = "PleIAs/SYNTH"


def dataset_files(revision: str | None = None) -> tuple[str, list[dict]]:
    if revision is None:
        with urllib.request.urlopen(f"https://huggingface.co/api/datasets/{REPO}") as response:
            revision = json.load(response)["sha"]
    with urllib.request.urlopen(
        f"https://huggingface.co/api/datasets/{REPO}/tree/{revision}?recursive=true&limit=1000"
    ) as response:
        files = json.load(response)
    parquet = sorted(
        (item for item in files if item.get("path", "").endswith(".parquet")),
        key=lambda item: item["path"],
    )
    if len(parquet) != 500:
        raise RuntimeError(f"Expected 500 SYNTH parquet shards, found {len(parquet)}")
    return revision, parquet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/synth_rag_exact_counts.json")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--limit-shards", type=int)
    args = parser.parse_args()

    revision, files = dataset_files()
    if args.limit_shards:
        files = files[: args.limit_shards]
    urls = [
        f"https://huggingface.co/datasets/{REPO}/resolve/{revision}/{item['path']}"
        for item in files
    ]
    db = duckdb.connect()
    db.execute(f"SET threads={args.threads}")
    started = time.time()
    rows = db.execute(
        """
        SELECT exercise, coalesce(language, 'missing') AS language,
               count(*) AS rows, sum(coalesce(words, 0)) AS words
        FROM read_parquet(?)
        GROUP BY ALL
        ORDER BY rows DESC
        """,
        [urls],
    ).fetchall()
    elapsed = time.time() - started
    total_rows = sum(row[2] for row in rows)
    rag_rows = [row for row in rows if row[0] == "rag"]
    report = {
        "dataset": REPO,
        "revision": revision,
        "parquet_shards": len(files),
        "parquet_bytes": sum(item.get("size", 0) for item in files),
        "rows": total_rows,
        "rag_rows": sum(row[2] for row in rag_rows),
        "rag_words": sum(row[3] for row in rag_rows),
        "rag_by_language": {row[1]: row[2] for row in rag_rows},
        "exercise_language_counts": [
            {"exercise": row[0], "language": row[1], "rows": row[2], "words": row[3]}
            for row in rows
        ],
        "elapsed_seconds": elapsed,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
