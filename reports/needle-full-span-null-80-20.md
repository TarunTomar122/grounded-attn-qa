# Full pretrained Needle span + NULL: 80/20 continuation

Date: 2026-08-17  
W&B: https://wandb.ai/tomartarun2001-adobe/grounded-attn-qa/runs/jhiu9tw1

## Checkpoint selection

The fixed full validation set contains 11,635 rows. The 70/30 checkpoints tied:

- `step-020000.pt`: threshold `-1.4453125`, all F1 `0.5409721580099253`, has-answer F1 `0.222884193048415`
- `best.pt`: threshold `-1.4453125`, all F1 `0.5409721580099253`, has-answer F1 `0.222884193048415`

The continuation started from `/workspace/runs/needle-full-decoder-span-null-70-30-v1/best.pt`.

## Training

The existing full span+NULL architecture and losses were continued with batch size 30, backbone LR `1e-5`, head LR `1e-4`, bf16, answerable fraction `0.8`, and max step `26000`. W&B synced 5 files.

Final step `26000`:

- Checkpoint: `/workspace/runs/needle-full-decoder-span-null-80-20-v1/step-026000.pt`
- Train loss `3.795828342437744`; start loss `1.7638863325119019`; end loss `2.0319418907165527`; grad norm `2.548375129699707`
- Full validation loss `3.843365730698576`
- Tuned threshold `-2.65625`; threshold F1 `0.5431098662028279`; has-answer F1 `0.23885470883478002`
- Full validation threshold EM `0.5253115599484315`; has-answer EM `0.2024604569420035`; no-answer accuracy `0.8343145500420521`

The 80/20 `best.pt` is the step-25000 checkpoint, whose full-validation threshold F1 was `0.545641033449852` and has-answer F1 was `0.2285647494181061`.

## Manual probe

Both checkpoints used the final tuned threshold `-2.65625`. The probe has two supported questions and one unsupported question.

| checkpoint | access code | opening date | unsupported |
| --- | --- | --- | --- |
| `step-026000.pt` | `QF-4302358-Y` | `3 March 2042` | refused: empty |
| `best.pt` (step 25000) | `QF-4302358-Y` | `3 March 2042` | refused: empty |

Exact inputs, raw spans, margins, thresholds, and predictions are saved at:

- `/workspace/runs/needle-full-decoder-span-null-80-20-v1/manual-final-step-026000.json`
- `/workspace/runs/needle-full-decoder-span-null-80-20-v1/manual-best.json`

