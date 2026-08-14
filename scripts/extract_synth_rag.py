from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import duckdb

from scripts.audit_synth_rag import REPO, dataset_files


REVISION = "0d6813a2966662c39f22f0b9af28a0c1c9f7a437"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract only English SYNTH RAG rows.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--limit-shards", type=int)
    args = parser.parse_args()

    revision, files = dataset_files(REVISION)
    urls = [
        f"https://huggingface.co/datasets/{REPO}/resolve/{revision}/{item['path']}"
        for item in files
    ]
    if args.limit_shards:
        urls = urls[: args.limit_shards]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    db = duckdb.connect()
    db.execute(f"SET threads={args.threads}")
    urls_sql = "[" + ",".join(f"'{url}'" for url in urls) + "]"
    output_sql = str(args.output).replace("'", "''")
    started = time.time()
    db.execute(
        f"""
        COPY (
            SELECT synth_id, query, constraints, synthetic_answer,
                   query_seed_url, seed_license, model, words
            FROM read_parquet({urls_sql})
            WHERE exercise = 'rag' AND language = 'en'
        ) TO '{output_sql}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 10000)
        """
    )
    rows = db.execute("SELECT count(*) FROM read_parquet(?)", [str(args.output)]).fetchone()[0]
    report = {
        "dataset": REPO,
        "revision": revision,
        "filter": "exercise = 'rag' AND language = 'en'",
        "rows": rows,
        "columns": [
            "synth_id",
            "query",
            "constraints",
            "synthetic_answer",
            "query_seed_url",
            "seed_license",
            "model",
            "words",
        ],
        "parquet": str(args.output),
        "parquet_bytes": args.output.stat().st_size,
        "sha256": sha256(args.output),
        "elapsed_seconds": time.time() - started,
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
