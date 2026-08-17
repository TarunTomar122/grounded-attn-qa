# Full pretrained Needle span + NULL: 80/20 continuation

Date: 2026-08-17

## Run

- Base: selected 70/30 checkpoint; full pretrained Needle encoder-decoder
- Composition: 80% answerable / 20% unanswerable
- Steps: 20,000 → 26,000
- Batch size: 30
- Optimizer rates: backbone `1e-5`, heads `1e-4`
- Precision: bfloat16
- Validation: unchanged fixed SQuAD2-derived set, 11,635 rows
- Checkpoints: `/workspace/runs/needle-full-decoder-span-null-80-20-v1/`
- W&B: [needle-full-decoder-span-null-squad2-80-20-v1](https://wandb.ai/tomartarun2001-adobe/grounded-attn-qa/runs/jhiu9tw1)

## 70/30 checkpoint selection

Both 70/30 checkpoints were evaluated on the same full validation set. They
tied on the selection metrics, so `best.pt` was used to resume the 80/20 run.

| Checkpoint | Tuned threshold | Threshold F1 | HasAns F1 | NoAns accuracy | False refusal | False answer |
|---|---:|---:|---:|---:|---:|---:|
| `step-020000.pt` | -1.4453125 | 0.5409721580099253 | 0.222884193048415 | 0.8454163162321279 | 0.7075571177504394 | 0.15458368376787215 |
| `best.pt` | -1.4453125 | 0.5409721580099253 | 0.222884193048415 | 0.8454163162321279 | 0.7075571177504394 | 0.15458368376787215 |

## Training metrics

| Step | Full validation loss | Threshold | Threshold F1 | HasAns F1 |
|---:|---:|---:|---:|---:|
| 23,000 | 3.955600028799981 | -3.015625 | 0.5415032154530123 | 0.20200174196762652 |
| 24,000 | 3.8828698304510607 | -3.1328125 | 0.5444628010824777 | 0.19662999834703385 |
| 25,000 | 3.889168991256006 | -2.923828125 | 0.545641033449852 | 0.2285647494181061 |
| 26,000 | 3.843365730698576 | -2.65625 | 0.5431098662028279 | 0.23885470883478002 |

Final step-26000 train loss was `3.795828342437744` (`start=1.7638863325119019`,
`end=2.0319418907165527`); gradient norm was `2.548375129699707`.

## Manual probes

The exact question/context/prediction JSONs are saved on the persistent volume:

- `/workspace/runs/needle-full-decoder-span-null-80-20-v1/manual-final-step-026000.json`
- `/workspace/runs/needle-full-decoder-span-null-80-20-v1/manual-best.json`

Both checkpoints used the final tuned threshold `-2.65625`. Both returned the
exact access code and opening date, and both abstained on the unsupported
question. The raw unsupported span was `East Annex`; thresholding correctly
returned an empty prediction.

| Checkpoint | Step | Access code | Opening date | Unsupported prediction | Unsupported null margin |
|---|---:|---|---|---|---:|
| `step-026000.pt` | 26,000 | exact | exact | empty | -1.7047357559204102 |
| `best.pt` | 25,000 | exact | exact | empty | -2.4117496013641357 |

W&B synced all five files. The pod was then stopped through RunPod and verified
with `desiredStatus=EXITED`.
