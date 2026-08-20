# Needle decoupled extraction + NULL: 10k result

Date: 2026-08-20

## What changed

Starting checkpoint: `reader-boost-90-10-step-046000.pt`.

The full pretrained Needle encoder-decoder was trained end-to-end with two pointer objectives:

1. Existing joint head: answer span versus one NULL option.
2. New independent source-only head: start/end positions over context tokens only.

The independent head was trained with weight `1.0`. For unanswerable SQuAD2 rows, its target was the original human plausible answer span when available. The joint head still learned NULL for those rows.

## Data audit

- Train rows: 125,563
- Validation rows: 11,635
- Train rows with auxiliary targets: 123,326
- Validation rows with auxiliary targets: 11,367
- Invalid auxiliary labels after mapping: 0
- Auxiliary labels outside their context window: 0

## Full validation results

The validation set stayed fixed at 11,635 rows.

| step | source-only forced F1 | source-only forced EM | raw joint F1 | threshold F1 | false refusal | false answer | no-answer accuracy |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2,000 | 56.4% | 40.3% | 61.8% | 55.7% | 66.0% | 16.8% | 83.2% |
| 4,000 | 58.1% | 41.9% | 61.3% | 55.7% | 74.2% | 11.7% | 88.3% |
| 6,000 | 59.3% | 42.9% | 61.8% | 55.6% | 69.5% | 15.1% | 84.9% |
| 8,000 | 60.0% | 43.5% | 62.1% | 55.9% | 60.1% | 20.5% | 79.5% |
| 10,000 | 60.8% | 44.4% | 62.3% | 55.7% | 62.2% | 19.3% | 80.7% |

Interpretation: the independent reader improved steadily. The model still refuses too many answerable questions after thresholding, while the raw span reader is considerably stronger than the final answer/refusal decision.

## Manual probe

Checkpoint: `step-010000.pt`.

### Input 1

Context: `Kifa's access code is AB-91827-X. Nuvora's access code is QF-4302358-Y. Torven's access code is LM-77301-P.`

Question: `What is Nuvora's access code?`

Expected: `QF-4302358-Y`

Model: `QF-4302358-Y`

### Input 2

Context: `The Lattice Ferry opened to public service on 3 March 2042 after its safety review. Its first prototype sailed in 2040.`

Question: `When did the Lattice Ferry open to public service?`

Expected: `3 March 2042`

Model: `3 March 2042`

### Input 3

Context: `The transfer record lists reference ST-92 and a destination in East Annex. It records no authorizing official.`

Question: `Which official authorized the sealed transfer?`

Expected: refuse

Raw span before the NULL threshold: `East Annex`

Final model output: refuse

## Artifacts

Remote checkpoint: `/workspace/experiments/decoupled-span-null-plausible-10k-v2/step-010000.pt`

Remote manual probe: `/workspace/experiments/decoupled-span-null-plausible-10k-v2/manual_probe_step-010000.json`

Remote W&B data is stored offline under `/workspace/wandb`. Cloud sync was unavailable during this run.

The earlier v1 run is excluded because its plausible-span mapping was invalid. This report covers only the corrected v2 run.
