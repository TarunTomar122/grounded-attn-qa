from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import duckdb
import torch

from grounded_qa.needle_tokenizer import EOS_ID, NeedleTokenizer
from grounded_qa.synth_rag import appears_unsupported, clean_answer, evidence_context


SOURCE_BUCKETS = (256, 512, 768, 1024)
TARGET_BUCKETS = (128, 256, 384, 512)


def bucket(length: int, boundaries: tuple[int, ...]) -> int:
    return next(boundary for boundary in boundaries if length <= boundary)


def split_for(value: str) -> str:
    return "validation" if int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "big") % 1000 < 10 else "train"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare length-bucketed N1 tensors from SYNTH RAG.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    tokenizer = NeedleTokenizer(args.tokenizer, append_markers=False)
    db = duckdb.connect()
    reader = db.execute(
        """
        SELECT synth_id, query, constraints, synthetic_answer, query_seed_url
        FROM read_parquet(?)
        """,
        [str(args.input)],
    ).fetch_record_batch(2048)
    grouped: dict[str, list[tuple[list[int], list[int]]]] = defaultdict(list)
    bucket_counts = defaultdict(int)
    stats = defaultdict(int)
    for batch in reader:
        for row in batch.to_pylist():
            stats["input_rows"] += 1
            answer = row["synthetic_answer"]
            if appears_unsupported(answer):
                stats["dropped_unsupported"] += 1
                continue
            cleaned = clean_answer(answer)
            if not cleaned:
                stats["dropped_empty_answer"] += 1
                continue
            context = evidence_context(
                row_id=row["synth_id"],
                query=row["query"],
                constraints=row["constraints"],
                answer=answer,
                tokenizer=tokenizer,
            )
            if context is None:
                stats["dropped_evidence"] += 1
                continue
            source_ids = tokenizer.encode_source(row["query"], context)
            target_ids = [*tokenizer.encode(cleaned), EOS_ID]
            if len(target_ids) > TARGET_BUCKETS[-1]:
                stats["dropped_target_length"] += 1
                continue
            split = split_for(str(row["query_seed_url"] or row["synth_id"]))
            source_bucket = bucket(len(source_ids), SOURCE_BUCKETS)
            target_bucket = bucket(len(target_ids), TARGET_BUCKETS)
            grouped[split].append((source_ids, target_ids))
            bucket_counts[f"{split}/s{source_bucket}/t{target_bucket}"] += 1
            stats[f"{split}_rows"] += 1
            stats[f"{split}_source_tokens"] += len(source_ids)
            stats[f"{split}_target_tokens"] += len(target_ids)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for split, rows in sorted(grouped.items()):
        source = torch.zeros((len(rows), SOURCE_BUCKETS[-1]), dtype=torch.int16)
        target = torch.zeros((len(rows), TARGET_BUCKETS[-1]), dtype=torch.int16)
        source_lengths = torch.empty(len(rows), dtype=torch.int16)
        target_lengths = torch.empty(len(rows), dtype=torch.int16)
        for index, (source_ids, target_ids) in enumerate(rows):
            source[index, : len(source_ids)] = torch.tensor(source_ids, dtype=torch.int16)
            target[index, : len(target_ids)] = torch.tensor(target_ids, dtype=torch.int16)
            source_lengths[index] = len(source_ids)
            target_lengths[index] = len(target_ids)
        path = args.output_dir / f"{args.input.stem}-{split}.pt"
        torch.save(
            {
                "source_ids": source,
                "target_ids": target,
                "source_lengths": source_lengths,
                "target_lengths": target_lengths,
            },
            path,
        )
        files.append({
            "path": str(path),
            "split": split,
            "rows": len(rows),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    report = {"input": str(args.input), "stats": dict(stats), "bucket_counts": dict(bucket_counts), "files": files}
    report_path = args.output_dir / f"{args.input.stem}-manifest.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
