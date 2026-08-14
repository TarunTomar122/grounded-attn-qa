from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb

from grounded_qa.needle_tokenizer import NeedleTokenizer
from grounded_qa.synth_rag import appears_unsupported, cited_source_ids, clean_answer, evidence_context, parse_sources


def percentiles(values: list[int]) -> dict[str, int]:
    if not values:
        return {}
    ordered = sorted(values)
    return {
        label: ordered[round((len(ordered) - 1) * fraction)]
        for label, fraction in (("p0", 0), ("p50", 0.5), ("p90", 0.9), ("p95", 0.95), ("p99", 0.99), ("max", 1))
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a deterministic sample of extracted SYNTH RAG.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-permille", type=int, default=25)
    args = parser.parse_args()
    if not 1 <= args.sample_permille <= 1000:
        parser.error("--sample-permille must be in [1, 1000]")

    db = duckdb.connect()
    total = db.execute("SELECT count(*) FROM read_parquet(?)", [str(args.input)]).fetchone()[0]
    rows = db.execute(
        """
        SELECT synth_id, query, constraints, synthetic_answer
        FROM read_parquet(?)
        WHERE hash(synth_id) % 1000 < ?
        """,
        [str(args.input), args.sample_permille],
    ).fetchall()
    tokenizer = NeedleTokenizer(args.tokenizer, append_markers=False)
    counts = {
        "sample_rows": len(rows),
        "parsed_sources": 0,
        "has_citations": 0,
        "missing_cited_source": 0,
        "appears_unsupported": 0,
        "full_context_fits_1024": 0,
        "target_fits_512": 0,
        "evidence_context_fits_1024": 0,
        "n1_answerable_usable": 0,
    }
    lengths = {name: [] for name in ("sources", "citations", "query_tokens", "full_source_tokens", "evidence_source_tokens", "raw_target_tokens", "clean_target_tokens")}
    for row_id, query, constraints, answer in rows:
        sources = parse_sources(constraints)
        citations = cited_source_ids(answer)
        source_ids = {source_id for source_id, _ in sources}
        query_tokens = len(tokenizer.encode(query))
        full_source_tokens = query_tokens + 1 + len(tokenizer.encode(constraints))
        raw_target_tokens = len(tokenizer.encode(answer)) + 1
        cleaned_answer = clean_answer(answer)
        clean_target_tokens = len(tokenizer.encode(cleaned_answer)) + 1
        unsupported = appears_unsupported(answer)
        selected = evidence_context(
            row_id=row_id,
            query=query,
            constraints=constraints,
            answer=answer,
            tokenizer=tokenizer,
        )
        selected_tokens = query_tokens + 1 + len(tokenizer.encode(selected)) if selected else 0
        counts["parsed_sources"] += bool(sources)
        counts["has_citations"] += bool(citations)
        counts["missing_cited_source"] += bool(set(citations) - source_ids)
        counts["appears_unsupported"] += unsupported
        counts["full_context_fits_1024"] += full_source_tokens <= 1024
        counts["target_fits_512"] += clean_target_tokens <= 512
        counts["evidence_context_fits_1024"] += selected is not None
        counts["n1_answerable_usable"] += selected is not None and clean_target_tokens <= 512 and not unsupported
        lengths["sources"].append(len(sources))
        lengths["citations"].append(len(citations))
        lengths["query_tokens"].append(query_tokens)
        lengths["full_source_tokens"].append(full_source_tokens)
        if selected_tokens:
            lengths["evidence_source_tokens"].append(selected_tokens)
        lengths["raw_target_tokens"].append(raw_target_tokens)
        lengths["clean_target_tokens"].append(clean_target_tokens)

    report = {
        "input": str(args.input),
        "total_rows": total,
        "sampling": f"hash(synth_id) % 1000 < {args.sample_permille}",
        "counts": counts,
        "rates": {
            key: value / max(len(rows), 1)
            for key, value in counts.items()
            if key != "sample_rows"
        },
        "estimated_n1_answerable_usable_rows": round(total * counts["n1_answerable_usable"] / max(len(rows), 1)),
        "lengths": {name: percentiles(values) for name, values in lengths.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
