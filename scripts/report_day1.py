from __future__ import annotations

import argparse
import json
import re
from statistics import median
from collections import Counter
from pathlib import Path


def json_lines(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    for line in path.open():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", default="artifacts/needleish26m_day1_report.md")
    args = parser.parse_args()
    root = Path(args.root)
    audit = json.loads((root / "artifacts/synth_audit.json").read_text())
    manifest = json.loads((root / "artifacts/synth_day1_parallel/manifest.json").read_text())
    benchmark = json.loads((root / "artifacts/foundation_benchmark.json").read_text())
    final_path = root / "runs/needleish26m-day1/final_metrics.json"
    final = json.loads(final_path.read_text()) if final_path.exists() else {}
    logs = json_lines(root / "runs/needleish26m-day1/launcher.log")
    train_curve = [row for row in logs if "train/loss_total" in row]
    val_curve = [row for row in logs if any(key.startswith("val/") for key in row)]
    probes = json_lines(root / "runs/needleish26m-day1/probes.jsonl")
    peak_reserved = max((row.get("system/gpu_reserved_gb", 0.0) for row in train_curve), default=0.0)
    benchmark_bf16 = [row for row in benchmark["results"] if row.get("precision") == "bf16" and not row.get("oom")]
    best_benchmark = max(benchmark_bf16, key=lambda row: row["tokens_per_sec"], default={})
    recent_rates = [row.get("system/tokens_per_sec", 0.0) for row in train_curve[-50:] if row.get("system/tokens_per_sec", 0.0) > 0]
    measured_tokens_per_sec = median(recent_rates) if recent_rates else best_benchmark.get("tokens_per_sec", 0.0)
    last_train = train_curve[-1] if train_curve else {}
    training_elapsed_estimate = (
        last_train.get("tokens_seen", 0.0) / measured_tokens_per_sec
        if measured_tokens_per_sec and last_train.get("tokens_seen")
        else None
    )
    final_for_report = {
        **final,
        "unique_rows_seen": last_train.get("unique_rows_seen", final.get("unique_rows_seen", 0)),
        "training_elapsed_seconds_estimate": training_elapsed_estimate,
        "evaluation_elapsed_seconds": final.get("elapsed_seconds"),
    }
    rows = []
    for line in (root / "artifacts/synth_day1_parallel/train.jsonl").open():
        rows.append(json.loads(line))
    validation_rows = []
    for line in (root / "artifacts/synth_day1_parallel/validation.jsonl").open():
        validation_rows.append(json.loads(line))
    train_urls = {row["source_url"] for row in rows}
    validation_urls = {row["source_url"] for row in validation_rows}
    exercise_counts = Counter(row["exercise"] for row in rows)
    overlap_counts = Counter("high" if row["answer_source_overlap"] >= 0.75 else "medium" if row["answer_source_overlap"] >= 0.25 else "low" for row in validation_rows)
    estimates = {
        f"{tokens / 1e9:g}B": (tokens / measured_tokens_per_sec / 3600) if measured_tokens_per_sec else None
        for tokens in (500_000_000, 1_000_000_000, 5_000_000_000)
    }
    report = {
        "architecture": benchmark["architecture"],
        "parameters": benchmark["parameters"],
        "dataset": {
            "name": "PleIAs/SYNTH",
            "audit": audit,
            "shards": manifest.get("parts"),
            "train_rows": len(rows),
            "validation_rows": len(validation_rows),
            "train_unique_sources": len(train_urls),
            "validation_unique_sources": len(validation_urls),
            "source_intersection": len(train_urls & validation_urls),
            "exercise_distribution": dict(exercise_counts),
            "validation_overlap_buckets": dict(overlap_counts),
            "manifest": manifest,
        },
        "benchmark": benchmark,
        "training": {
            "final": final_for_report,
            "train_curve_points": train_curve,
            "validation_curve_points": val_curve,
            "probe_summaries": [{"tokens_seen": row.get("tokens_seen"), **row.get("summary", {})} for row in probes],
            "peak_reserved_gb_from_training_log": peak_reserved,
            "measured_tokens_per_sec": measured_tokens_per_sec,
            "training_elapsed_seconds_estimate": training_elapsed_estimate,
            "mps_benchmark_path": "benchmarks/MPS_2026-08-13.md" if (root / "benchmarks/MPS_2026-08-13.md").exists() else None,
            "estimated_hours": estimates,
        },
    }
    json_path = Path(args.output).with_suffix(".json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    markdown = f"""# Needleish26M Foundation Pretraining — Day 1

## Result

The run ended at `{final_for_report.get('tokens_seen', 'unknown')}` total source+target token exposures. Estimated training wall time was `{training_elapsed_estimate:.0f}` seconds from the steady-state training log; the exact-checkpoint evaluation took `{final.get('elapsed_seconds', 'unknown')}` seconds. The report is evidence-only; no downstream SQuAD, pointer, refusal, SFT, or RL training was run.

## Architecture

```json
{json.dumps({'parameters': benchmark['parameters'], **benchmark['architecture']}, indent=2)}
```

Parameter breakdown: `{json.dumps(benchmark.get('breakdown', {}), sort_keys=True)}`.

## Data

- Dataset: PleIAs/SYNTH, English-only after filtering.
- Shards: `{manifest.get('parts')}`.
- Train/validation rows: `{len(rows)}` / `{len(validation_rows)}`.
- Unique source URLs: `{len(train_urls)}` / `{len(validation_urls)}`.
- Global source URL intersection: `{len(train_urls & validation_urls)}`.
- Train exercise distribution after the deterministic cap: `{dict(exercise_counts)}`.
- Validation answer-overlap buckets: `{dict(overlap_counts)}`.
- Source/target limits: 512 / 256 tokens.

## Learning and grounding

- Final metrics: `{json.dumps(final_for_report, sort_keys=True)}`.
- Training curve points: `{len(train_curve)}`.
- Validation curve points: `{len(val_curve)}`.
- Qualitative probe checkpoints: `{len(probes)}`.
- Peak training reserved GPU memory: `{peak_reserved:.2f} GB`.
- Measured steady-state training throughput: `{measured_tokens_per_sec:.0f}` source+target tokens/sec.
- M2 Max benchmark: `benchmarks/MPS_2026-08-13.md`.
- Context-dependency gap is recorded in each validation point as `val/context_dependency_gap`.

## Throughput estimates

At the measured throughput, estimated wall-clock time is `{json.dumps(estimates)}` hours for 500M, 1B, and 5B tokens.

## Recommendation

Use the curves above to decide whether to scale the same foundation run tomorrow. Keep downstream specialization out of this decision.
"""
    Path(args.output).write_text(markdown)
    print(json.dumps({"markdown": str(args.output), "json": str(json_path), "source_intersection": len(train_urls & validation_urls), "final": final}, indent=2))


if __name__ == "__main__":
    main()
