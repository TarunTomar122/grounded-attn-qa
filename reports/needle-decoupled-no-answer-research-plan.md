# Overnight follow-up: independent NULL supervision

## Decision

The 10k decoupled extraction run improved source-only localization from 56.4% to 60.8% forced F1, while thresholded answer/refusal F1 stayed flat at 55.6–55.9%. This means the new extraction head is learning, but its scores do not affect the final NULL-versus-span decision.

The next run changes one thing: independently supervise the existing NULL score as a binary answerability signal.

```text
null_score = joint_start_NULL_logit + joint_end_NULL_logit
null_target = 1 for unanswerable, 0 for answerable

loss = joint_span_NULL_loss
     + source_only_extraction_loss
     + BCEWithLogitsLoss(null_score, null_target)
```

The architecture, data, checkpoint, learning rates, batch size, precision, threshold procedure, and extraction loss remain unchanged.

## Why this change

Read+Verify reports the same separation: an independent plausible-span loss and an independent no-answer sigmoid loss are added because shared normalization makes span confidence and no-answer confidence compete. The relevant original paper is [Read + Verify](https://ojs.aaai.org/index.php/AAAI/article/download/4619/4497).

The follow-up therefore tests the smallest change aimed at the actual observed bottleneck: NULL calibration.

## Diagnostics

Alongside existing metrics, the run logs:

- answerability ROC AUC and average precision;
- NULL-logit means for answerable and unanswerable rows;
- p10/p50/p90 margins for both classes;
- existing raw span, thresholded F1, false-refusal, false-answer, and no-answer accuracy.

## Decision criteria at 10k

Success requires all of:

- threshold F1 at least 57.7%;
- false refusal at most 52.2%;
- false answer at most 22.3%;
- raw joint F1 at least 61.3%;
- source-only forced F1 at least 59.8%.

If threshold F1 improves by less than one point, or the gain only swaps refusals for false answers, stop this line rather than extending it blindly.

## Provenance

- Valid source run: `reports/needle-decoupled-span-null-plausible-10k-v2.md`.
- Starting checkpoint: `reader-boost-90-10-step-046000.pt`.
- The earlier v1 run is excluded because its plausible-span mapping was invalid.
