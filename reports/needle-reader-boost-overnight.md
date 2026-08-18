# Needle reader boost: overnight results

All metrics below are from the full 11,635-row validation set unless noted.

## Runs

| Stage | Starting checkpoint | Composition | End | Raw HasAns F1 | Threshold HasAns F1 | False refusal | False answer | NoAns accuracy |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Reader boost | 80/20 step 26,000 | answerable-only, NULL excluded | 36,000 | 61.14% | — | — | — | — |
| Calibration | reader step 36,000 | 70% answerable / 30% unanswerable | 42,000 | 60.97% | **29.92%** | **61.85%** | 20.72% | 79.28% |
| Calibration | 70/30 step 42,000 | 90% answerable / 10% unanswerable | 46,000 | **61.88%** | 27.74% | 65.94% | **18.17%** | **81.83%** |

Threshold metrics use the checkpoint's tuned NULL threshold. Raw HasAns metrics force the best non-NULL span and therefore measure the reader separately from refusal calibration.

## What happened

- The answer reader improved from the 26k 80/20 starting point and reached 61.14% raw HasAns F1 at the end of the answerable-only phase.
- The 70/30 phase gave the best answer/refusal balance tested: thresholded HasAns F1 rose to 29.92% and false refusal fell to 61.85%.
- The 90/10 phase slightly improved the raw reader to 61.88%, and reduced false answers to 18.17%, but it accepted fewer valid answers than 70/30. Its thresholded HasAns F1 fell to 27.74% and false refusal rose to 65.94%.
- NULL behavior remains strong: even the 90/10 endpoint correctly rejects 81.83% of unanswerable validation examples.

## Checkpoints and W&B

- Reader boost: `/workspace/runs/needle-reader-boost-answerable-v1/step-036000.pt`
- 70/30: `/workspace/runs/needle-reader-boost-70-30-v1/step-042000.pt`
- 90/10: `/workspace/runs/needle-reader-boost-90-10-v1/step-046000.pt`
- Reader W&B: https://wandb.ai/tomartarun2001-adobe/grounded-attn-qa/runs/w2y31mzr
- 70/30 W&B: https://wandb.ai/tomartarun2001-adobe/grounded-attn-qa/runs/sm8l1gif
- 90/10 W&B: https://wandb.ai/tomartarun2001-adobe/grounded-attn-qa/runs/ymiyhjwr

## Decision

Keep both checkpoints:

- Use 70/30 step 42,000 as the best current selective QA checkpoint.
- Use 90/10 step 46,000 as the best raw-reader checkpoint among these stages, with a more conservative answer/refusal tradeoff.

This run did not solve the task yet. It showed that more reader training helps the forced reader modestly, while answerable-heavy calibration alone does not eliminate excessive false refusal.
