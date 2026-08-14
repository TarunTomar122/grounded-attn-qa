#!/usr/bin/env bash
set -euo pipefail

repo_dir="/root/autodl-tmp/grounded-attn-qa"
cd "$repo_dir"
source .venv/bin/activate
export PYTHONPATH="$repo_dir${PYTHONPATH:+:$PYTHONPATH}"

sleep 600

for part in 0 1 2 3; do
  while [[ ! -s "artifacts/synth_parts/$part/manifest.json" ]]; do
    sleep 30
  done
done

combined_dir="artifacts/synth_day1_parallel"
mkdir -p "$combined_dir"
python - <<'PY'
import json
from pathlib import Path

root = Path("artifacts/synth_parts")
out = Path("artifacts/synth_day1_parallel")
manifests = [json.loads((root / str(index) / "manifest.json").read_text()) for index in range(4)]
if not all(manifest["source_disjoint"] for manifest in manifests):
    raise SystemExit("a shard manifest failed its source-disjoint check")
cap = 3000
train_counts = {}
validation_counts = {}
train_urls = set()
validation_urls = set()
with (out / "train.jsonl").open("w") as train, (out / "validation.jsonl").open("w") as validation:
    for index in range(4):
        with (root / str(index) / "train.jsonl").open() as source:
            for line in source:
                row = json.loads(line)
                exercise = row["exercise"]
                if train_counts.get(exercise, 0) >= cap:
                    continue
                train.write(line)
                train_counts[exercise] = train_counts.get(exercise, 0) + 1
                train_urls.add(row["source_url"])
        with (root / str(index) / "validation.jsonl").open() as source:
            for line in source:
                row = json.loads(line)
                validation.write(line)
                validation_counts[row["exercise"]] = validation_counts.get(row["exercise"], 0) + 1
                validation_urls.add(row["source_url"])
intersection = train_urls & validation_urls
if intersection:
    raise SystemExit(f"source leakage detected: {len(intersection)} URLs")
summary = {
    "dataset": "PleIAs/SYNTH",
    "parts": [manifest["shards"] for manifest in manifests],
    "source_disjoint": all(manifest["source_disjoint"] for manifest in manifests) and not intersection,
    "train_exercise_distribution_before_cap": {
        exercise: sum(manifest["exercise_distribution"]["train"].get(exercise, 0) for manifest in manifests)
        for exercise in {exercise for manifest in manifests for exercise in manifest["exercise_distribution"]["train"]}
    },
    "train_exercise_distribution_after_cap": train_counts,
    "validation_exercise_distribution": validation_counts,
    "train_exercise_cap": cap,
    "counts": {
        "train": sum(train_counts.values()),
        "validation": sum(validation_counts.values()),
    },
    "unique_source_urls": {
        split: sum(manifest["unique_source_urls"][split] for manifest in manifests)
        for split in ("train", "validation")
    },
}
(out / "manifest.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2), flush=True)
PY

mkdir -p runs/needleish26m-day1
export PYTHONUNBUFFERED=1
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  wandb_args=(--wandb)
else
  wandb_args=()
fi
exec python scripts/train_foundation.py \
  --data-dir "$combined_dir" \
  --tokenizer artifacts/needleish26m/needle.model \
  --output-dir runs/needleish26m-day1 \
  --batch-size 8 \
  --grad-accum 1 \
  --workers 4 \
  --max-tokens 250000000 \
  --max-hours 2 \
  --device cuda \
  --precision bf16 \
  "${wandb_args[@]}"
