# A2b diagnostic follow-up

## 64-example SQuAD overfit

This was a read-only diagnostic from `runs/phase-a1d-500/latest.pt` with a fresh A2b start head, 64 fixed answerable SQuAD validation rows, fp32 MPS, and the A2b losses (`lambda_pointer=1`, `lambda_start=1`, first-pointer weight `4`). It did not modify a checkpoint.

| Step | Total loss | Encoder start accuracy | First-pointer accuracy |
| ---: | ---: | ---: | ---: |
| 0 | 33.5041 | 0.0% | 0.0% |
| 50 | 10.1264 | 23.4% | 18.8% |
| 100 | 2.7171 | 65.6% | 62.5% |
| 250 | 0.1571 | 98.4% | 98.4% |
| 500 | 0.0208 | 100.0% | 100.0% |

Conclusion: the head, aligned source positions, and attention-only representation can fit the task. The A2b failure is transfer/generalization on varied SQuAD contexts, not a basic capacity or alignment failure.

## Research implication

Classic extractive QA systems make question–context interaction explicit before predicting start/end positions ([BiDAF](https://arxiv.org/abs/1611.01603), sections 4–6). Recent controlled attention-only results report that attention-only models can be strong on context-grounded answers, while their remaining deficit concentrates on low-context query prediction ([A Controlled Study of Attention-Only Transformers](https://arxiv.org/abs/2607.18363)).

The next safe experiment is an easier natural-QA curriculum with short contexts/answers plus replay, keeping the architecture unchanged. It tests whether the model needs a difficulty ramp before an architecture change.

## Easy-curriculum pilot

Starting from `runs/phase-a2b-1000/latest.pt`, the SQuAD share was filtered to 3,664 rows with context length ≤256 BPE tokens and answer length ≤6 BPE tokens. The same 65/25/10 SQuAD/A1/A1d mixture ran for 500 steps with a reset AdamW optimizer. It produced `runs/phase-a2c-easy-500/latest.pt`.

| Metric | A2b @1000 | Easy pilot @500 |
| --- | ---: | ---: |
| Full SQuAD loss | 8.230 | 10.389 |
| Full SQuAD start-head accuracy | 5.57% | 6.51% |
| Full SQuAD teacher-forced pointer accuracy | 67.1% | 36.0% |
| Fixed 128-row generated F1 | 5.20% | 10.42% |
| Fixed 128-row generated first-start accuracy | 7.81% | 10.16% |

Conclusion: the curriculum helped a narrow generated probe but caused broad teacher-forced regression. Do not promote this checkpoint or continue it without a replay/validation redesign.
