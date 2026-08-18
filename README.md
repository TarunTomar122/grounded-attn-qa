# Grounded attention-only pointer-generator

> **Current program (2026-08-14):** adapt the released pretrained Needle 26M
> checkpoint into a grounded RAG model, then compare matched generative and
> pointer/provenance variants. The measured audit and live decisions are in
> [NEEDLE_RAG_LOG.md](NEEDLE_RAG_LOG.md). The scratch-pretraining material below
> is retained as experiment history.

This repository is the fresh redo of the grounded QA experiment. The canonical
protocol is in [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md). The model is an
approximately 23.21M parameter encoder-decoder with no Transformer FFNs, RoPE,
RMSNorm, a context-only pointer-generator, and a separate answerability head.

The latest restart is the 26.24M-parameter `needleish26m_v1` foundation model.
Its Day 1 run completed 150,003,051 token exposures on the source-disjoint
PleIAs/SYNTH split; architecture, data composition, curves, metrics, and manual
probes are in [the Day 1 report](artifacts/needleish26m_day1_report.md).

Current implementation status:

- [x] 6-layer bidirectional encoder + 6-layer causal decoder
- [x] No Transformer FFN/MLP blocks
- [x] RMSNorm and RoPE
- [x] 32k byte-level BPE tokenizer training
- [x] Context-only pointer distribution and copy/generate gate
- [x] Strict Phase A copy-only mode with learned EOS termination
- [x] Separate answerability loss path
- [x] Synthetic Phase A generator with held-out prefixes/templates
- [x] Checkpoint/resume, MPS telemetry, gradient accumulation, W&B hooks
- [x] Focused tests and 64-row overfit gate
- [x] 250-step MPS benchmark: [benchmarks/MPS_2026-08-13.md](benchmarks/MPS_2026-08-13.md)
- [ ] Full Phase A training
- [ ] SQuAD/NQ Phase B data adapters
- [ ] CoQA/MS MARCO Phase C data adapters
- [ ] Refusal/RAG phases and adversarial evaluation

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Keep the W&B credential outside the repository:

```bash
export WANDB_API_KEY='your-key'
```

## Tokenizer and gates

Train the tokenizer only on Phase A training text:

```bash
python scripts/train_tokenizer.py \
  --output artifacts/tokenizer.json \
  --train-n 200000
```

Run the mandatory tiny overfit check:

```bash
python scripts/overfit_64.py \
  --tokenizer artifacts/tokenizer.json \
  --steps 2000 \
  --batch-size 8
```

Run the measured MPS benchmark:

```bash
python scripts/benchmark_mps.py \
  --steps 250 \
  --batch-sizes 2,4,8,16 \
  --precisions fp32,fp16,bf16
```

## Phase A

The Phase A entrypoint is ready. It uses the measured FP32 starting choice,
microbatch 16, accumulation 2, cosine decay, warmup, answer-only sequence loss,
and checkpointing every 250/500-step boundary:

```bash
WANDB_API_KEY="$WANDB_API_KEY" python scripts/train.py \
  --phase A \
  --tokenizer artifacts/tokenizer.json \
  --train-n 200000 \
  --val-n 10000 \
  --steps 4000 \
  --batch-size 16 \
  --grad-accum 2 \
  --precision fp32 \
  --wandb
```

The script prints the exact parameter breakdown and measured tokens/sec. Do not
start Phase B until the Phase A gate in the plan is satisfied.

The first strict-copy probe is documented in [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md)
and synced as [W&B run hai3wnwa](https://wandb.ai/tomartarun2001-adobe/grounded-attn-qa/runs/hai3wnwa).

The old `train.py`, `data.py`, and released causal checkpoint are retained only
as legacy artifacts from the previous experiment; all redo commands use the
`grounded_qa/` package and `scripts/` entrypoints.

## Browser demo

Try the current grounded reader in the [Needle browser demo](https://taruntomar122.github.io/grounded-attn-qa/).
The GitHub Page is a static frontend; the checkpoint stays on the inference
machine. Start the demo server with the latest full span/NULL checkpoint:

```bash
python scripts/serve_needle_demo.py \
  --checkpoint /workspace/runs/needle-reader-boost-90-10-v1/step-046000.pt \
  --tokenizer /workspace/needle26-public/tokenizer.model
```

Then open the page with `?api=http://127.0.0.1:8000`, or enter that API URL in
the page. You can paste arbitrary context and questions; the server returns an
answer copied from the context or a refusal.
