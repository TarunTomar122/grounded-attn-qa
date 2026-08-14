# Grounded Attention-Only Pointer-Generator

This is the canonical redo protocol. The current implementation is the Phase A
foundation; later phases must be added without silently changing this contract.

## Hypothesis

Removing Transformer FFN/MLP blocks may reduce parametric factual storage and
bias the network toward evidence already present in the current context. This
is a hypothesis, not a guarantee that attention-only models cannot memorize.

The model is a task-specific grounded QA system, not a general language model.

## Main model: `attn_pg_23m_v1`

- Encoder-decoder pointer-generator, trained from random initialization.
- 6 bidirectional encoder layers and 6 autoregressive decoder layers.
- `d_model=384`, 6 heads, 64 dimensions/head.
- No Transformer FFN/MLP in either stack.
- RMSNorm (`eps=1e-5`), dropout `0.10`, RoPE (`theta=10000`).
- Byte-level BPE tokenizer trained from scratch with 32,000 vocabulary items.
- Shared encoder/decoder embeddings and tied vocabulary projection.
- Source length 512 initially, 1024 for Phase E; target length 64.
- Context-only pointer mask; question, special tokens, and padding cannot be copied.
- Separate `<CLS>` answerability head. It is frozen during Phase A v2; negative
  rows receive answerability loss only in the later refusal phases.
- Planned parameter count is about 23.21M. Current implementation reports
  23,210,115 parameters including the Phase A EOS stop head.

Input:

```text
<CLS> <Q> question <SEP> <CTX> context <EOS>
```

The later full model distribution is:

```text
P_final = p_gen * P_vocab + (1 - p_gen) * P_copy
```

`P_copy` is a pointer distribution over context positions, summed by token ID
when a token occurs more than once.

Phase A v2 is deliberately stricter: it disables the vocabulary distribution,
ignores `p_gen`, freezes answerability, and learns only context pointer tokens
plus EOS termination. Synthetic answers are generated from a large random
space, and a byte-level BPE boundary check requires every target token span to
occur in the context.

## Training curriculum

Every phase starts from the best checkpoint of the previous phase and resets
the optimizer and scheduler. Early debugging uses seed 42 only.

| Phase | Purpose | Data | Steps |
|---|---|---|---:|
| A | Learn unfamiliar evidence copying | 200k synthetic train / 10k val / 10k test | 4k |
| B | Real reading comprehension | answerable SQuAD 2.0 + short-answer Natural Questions | 8k |
| C | Grounded paraphrasing | CoQA + filtered MS MARCO + Phase B extractive rows | 6k |
| D | Refusal and answerability calibration | staged negatives from cross-document, SQuAD 2, NQ, swaps, numeric/date errors, conflicts | 8k |
| E | Deployment-shaped multi-chunk RAG | 70% answerable / 30% unanswerable packs, source length 1024 | 4k |

Total planned optimizer steps: approximately 30k.

### Phase A gate

Do not proceed until the synthetic model reaches token F1 >= 0.97 and EM >=
0.90, with held-out template accuracy tracked separately. If EM is below 0.70
after 8k total Phase A steps, stop and debug the implementation.

The first strict-copy probe ran for 1,000 steps with 200k train rows, 1k rows
per fixed validation split, and 64 generated examples per split:

| Split | Generated EM | Pointer token accuracy | Result |
|---|---:|---:|---|
| Same templates / unseen entities | 90.6% | 93.5% | Passes literal copy |
| Unseen templates / familiar entities | 9.4% | 18.9% | Fails language generalization |
| Unseen templates / unseen entities + hard distractors | 0.0% | 17.7% | Fails robust retrieval |
| Random access codes | 0.0% | 0.0% | Fails retrieval probe |

This is a useful diagnostic failure, not a Phase A pass. Do not start SFT,
RL, or real-data phases from this checkpoint.

### Refusal gate

After Phase D, calibrate the answerability threshold on a held-out split. Select
the lowest threshold with false-answer rate <= 2%; aim for <= 1% later. Report
answer coverage, false refusal, false answer, precision, AUROC, and AUPRC.

## Data rules

- Canonical row fields: `id`, `question`, `context`, `answer`, `answerable`,
  `evidence`, `source`, `phase`, `hardness`, and `metadata`.
- Split by document/passage where the source dataset supports it.
- Deduplicate normalized question/context pairs and exact contexts where practical.
- Keep synthetic entities and templates separate across train/validation/test.
- Maintain an untouched 200-example adversarial suite.
- Natural sources: SQuAD 2.0, Natural Questions, CoQA, and filtered MS MARCO.
- QASPER is reserved for out-of-domain evaluation, not initial training.
- Reject MS MARCO targets whose numeric facts are unsupported by the selected passages.

## MPS procedure

Before proper Phase A training:

1. Fit 64 synthetic examples repeatedly; require >98% token accuracy.
2. Run 250 optimizer steps for candidate microbatches 2, 4, 8, and 16 at source length 512.
3. Compare FP32 with stable MPS mixed precision; record tokens/sec, step time,
   memory, loss, and NaN/Inf count.
4. Choose the largest stable microbatch and use gradient accumulation to target
   effective batch 32.

Do not assume CUDA AMP behavior or hard-code runtime estimates on Apple Silicon.

## W&B

- Project: `grounded-attn-qa`.
- Main group: `attnonly-main`.
- Phase names: `attn23m-s42-A-copy` through `attn23m-s42-E-rag`.
- Use `WANDB_API_KEY` from the environment; never put the credential in the
  repository or logs.
- Log losses, learning rate, gradient norm, throughput, MPS memory, dataset
  breakdowns, pointer copy/generate behavior, answerability distributions,
  threshold curves, fixed probe tables, and pointer heatmaps.

## Baselines and ablations, only after the main pipeline works

- Parameter-matched FFN Transformer: `d_model=288`, 6+6 layers, `d_ff=1152`.
- Width-matched FFN Transformer: `d_model=384`, 6+6 layers, `d_ff=1536`.
- Remove pointer; force copy-only; remove paraphrase phase; remove refusal curriculum.

Use identical datasets, tokenizer, evaluation, and training budget for the main
comparison. Never fabricate comparison plots or change ratios to rescue a run.
