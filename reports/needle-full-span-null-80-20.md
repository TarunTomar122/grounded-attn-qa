# Full pretrained Needle span + NULL: 80/20 continuation

Date: 2026-08-17

## Run

- Base: released pretrained Needle checkpoint (`model.safetensors`), full
  encoder-decoder trainable
- Resume: selected 70/30 `best.pt`; it tied `step-020000.pt` on the fixed full
  validation set
- Composition: 80% answerable / 20% unanswerable; batch size 30
- Steps: 20,000 → 26,000
- Optimizer rates: backbone `1e-5`, heads `1e-4`
- Precision: bfloat16
- Validation: unchanged SQuAD2-derived set, 11,635 rows
- Checkpoints: `/workspace/runs/needle-full-decoder-span-null-80-20-v1/`
- W&B: [needle-full-decoder-span-null-squad2-80-20-v1](https://wandb.ai/tomartarun2001-adobe/grounded-attn-qa/runs/jhiu9tw1)

## 70/30 checkpoint selection

| Checkpoint | Threshold | Threshold F1 | HasAns F1 |
|---|---:|---:|---:|
| `step-020000.pt` | -1.4453125 | 0.5409721580099253 | 0.222884193048415 |
| `best.pt` | -1.4453125 | 0.5409721580099253 | 0.222884193048415 |

The records tie; the continuation used `/workspace/runs/needle-full-decoder-span-null-70-30-v1/best.pt`.

## Final 80/20 metrics

The final full validation at step 26,000 used tuned threshold `-2.65625`:

- Validation loss: `3.843365730698576`
- Threshold F1: `0.5431098662028279`
- Threshold HasAns F1: `0.23885470883478002`
- Threshold EM: `0.5253115599484315`
- Threshold HasAns EM: `0.2024604569420035`
- False-refusal rate: `0.690158172231986`
- False-answer rate: `0.16568544995794784`
- NoAns accuracy: `0.8343145500420521`
- Raw HasAns F1: `0.5838716547258801`

Final quick/training record: quick loss `3.8592029677497015`, quick threshold
F1 `0.5742515297202797`, quick HasAns F1 `0.15751822645439667`, train loss
`3.795828342437744`, start loss `1.7638863325119019`, end loss
`2.0319418907165527`, grad norm `2.548375129699707`.

W&B reported the run synced after the final checkpoint.

## Manual probes

The existing `scripts/manual_probe_full_span_null.py` ran both checkpoints with
threshold `-2.65625`. Each JSON contains the exact question, context, expected
answer, raw span, prediction, NULL margin, and threshold:

- `/workspace/runs/needle-full-decoder-span-null-80-20-v1/final-step-026000-probe.json`
- `/workspace/runs/needle-full-decoder-span-null-80-20-v1/best-probe.json`

| Checkpoint | Access code | Opening date | Unsupported |
|---|---|---|---|
| `step-026000.pt` | `QF-4302358-Y` | `3 March 2042` | abstained; raw span `East Annex` |
| `best.pt` (step 25,000) | `QF-4302358-Y` | `3 March 2042` | abstained; raw span `East Annex` |

Both probes answered the two supported questions and abstained on the
unsupported question. The run directory is on the persistent remote volume;
only this report was added to the repository.
