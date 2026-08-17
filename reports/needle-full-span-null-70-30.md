# Full pretrained Needle span + NULL: 70/30 continuation

Date: 2026-08-17

## Purpose

Continue the validated full pretrained Needle encoder-decoder grounded QA run
with more answerable examples, because the 50/50 curriculum learned NULL much
faster than exact answer spans.

## Run

- Base: released pretrained Needle checkpoint (`model.safetensors`), full
  encoder-decoder trainable
- Task: context-grounded span extraction with a learned NULL span for
  unanswerable questions
- Resume: 50/50 run at step 14,000
- Composition: 70% answerable / 30% unanswerable; batch size 30 means 21/9
  examples per batch, deterministically sampled and shuffled
- Steps: 14,000 → 20,000 (6,000 continuation steps)
- Optimizer rates: backbone `1e-5`, heads `1e-4`
- Precision: bfloat16
- Evaluation: quick 512-row checks every 500 steps; full fixed validation
  every 2,000 steps
- Validation: unchanged SQuAD2-derived set, 11,635 rows
- Checkpoints: `/workspace/runs/needle-full-decoder-span-null-70-30-v1/`
- W&B: [needle-full-decoder-span-null-squad2-70-30-v1](https://wandb.ai/tomartarun2001-adobe/grounded-attn-qa/runs/ie5wnlft)

## Full validation results

Percentages are easier to read; all values come from the trainer's full
11,635-row evaluation.

| Metric | 50/50 at 14k | 70/30 at 16k | 70/30 at 18k | 70/30 at 20k |
|---|---:|---:|---:|---:|
| Validation loss | 3.7102 | 3.7761 | 3.7797 | **3.6957** |
| Raw HasAns F1 | 53.78% | 54.55% | 55.47% | **56.06%** |
| Raw HasAns EM | 37.66% | 38.19% | 39.00% | **39.88%** |
| HasAns start accuracy | 20.76% | 27.19% | **28.44%** | 28.37% |
| HasAns end accuracy | 20.77% | 27.07% | **29.10%** | 27.68% |
| Thresholded HasAns F1 | 13.21% | 17.04% | 19.23% | **21.77%** |
| False-refusal rate | 83.29% | 77.64% | 75.29% | **71.58%** |
| False-answer rate | 8.34% | 11.05% | 12.88% | 14.97% |
| NoAns accuracy | 91.66% | 88.95% | 87.12% | 85.03% |

## What changed

The answerable-heavy continuation did what it was intended to do: the model
became less likely to refuse answerable questions. False refusal fell from
83.29% to 71.58%, while thresholded answer F1 rose from 13.21% to 21.77%.
The tradeoff was lower NoAns accuracy and more unsupported answers: false
answers rose from 8.34% to 14.97%.

The reader itself also improved: raw HasAns F1 rose from 53.78% to 56.06%.
This means the remaining limitation is both localization quality and the
answer-versus-NULL decision boundary; it is not only a sampling problem.

## Saved artifacts

- Final checkpoint: `step-020000.pt` (remote RunPod volume)
- Selected checkpoint: `best.pt` (remote RunPod volume)
- Earlier continuation checkpoints: steps 15k–19k

The remote run completed and W&B reported all five files synced.
