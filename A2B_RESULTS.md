# A2b — Answer-start localization

Run: [`attn23m-s42-A2b-start-head-v1-1000`](https://wandb.ai/tomartarun2001-adobe/grounded-attn-qa/runs/kr8jxqyh)

Configuration:

- Starting checkpoint: `runs/phase-a1d-500/latest.pt`
- 1,000 steps, reset optimizer, MPS fp32
- 65% answerable SQuAD 2.0, 25% A1 replay, 10% A1d shared-prefix replay
- `lambda_pointer = 1.0`, `lambda_start = 1.0`, first-pointer weight `4.0`
- 23,210,884 parameters; the auxiliary `Linear(384 -> 1)` start head adds 385

## A2a vs A2b at 1,000 steps

| Metric | A2a baseline | A2b |
| --- | ---: | ---: |
| SQuAD validation loss | 1.289 | 8.230 |
| Teacher-forced pointer accuracy | 74.3% | 67.1% |
| Encoder start-head accuracy | N/A | 5.6% |
| Decoder first-pointer position accuracy | 2.7% | 7.8% |
| Generated token F1 | 7.1% | 5.2% |
| Generated greedy EM | 0.6% | 0.0% |
| A1 familiar EM retention | not logged | 48.4% |
| A1b entity EM retention | not logged | 46.1% |
| A1d shared-prefix EM retention | not logged | 96.1% |

The generated metrics use a fixed 128-row probe; loss and teacher-forced metrics use all 5,849 answerable validation rows.

## Result

A2b is a partial negative result. Explicit start supervision improved decoder first-pointer accuracy over A2a, but the encoder start head reached only 5.6%, below the 25–30% success threshold. Synthetic retention stayed healthy, especially A1d at 96.1% EM and 98.4% gold-start accuracy.

The next decision should come from research/diagnosis rather than another unbounded run. The Chrome ChatGPT handoff is pending sign-in in the requested browser profile; the local tab was left open for that handoff.
