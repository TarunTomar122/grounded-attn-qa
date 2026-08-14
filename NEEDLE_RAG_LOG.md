# Needle 26M grounded RAG log

## 2026-08-14 — starting-point audit

The available public checkpoint is the function-call post-trained Needle model,
not a verified clean foundation checkpoint. Its untouched free-running output
on five QA probes (correct, wrong, empty, and counterfactual contexts) was always:

```text
<tool_call> []</s>
```

This makes restoration of contextual question answering the purpose of N1.

A frozen 256-question answerable SQuAD2 validation slice produced the same
result across 776 free-running inputs:

| Condition | Rows | EM | Token F1 | EOS | Output changed from correct context |
|---|---:|---:|---:|---:|---:|
| Correct evidence | 256 | 0% | 0% | 100% | — |
| Wrong evidence | 256 | 0% | 0% | 100% | 0% |
| Empty evidence | 256 | 0% | 0% | 100% | 0% |
| Counterfactual evidence | 8 | 0% | 0% | 100% | — |

Every prediction was the same two-token payload, `<tool_call> []`, followed by
EOS. N0 therefore has zero measurable context dependence under raw greedy QA
decoding.

### Checkpoint

| Item | Verified value |
|---|---|
| Hugging Face repository | `Cactus-Compute/needle` |
| Repository revision | `5f89b4307696d669c3df1d38ae057e6e1728b107` |
| Canonical historical implementation | `cactus-compute/needle@a5af1f8282c3aa56d6fbfde77648146114ad31fa` |
| Language-model parameters | 26,233,372 |
| Encoder / decoder layers | 12 / 8 |
| Hidden size | 512 |
| Query / KV heads | 8 / 4 |
| Vocabulary | 8,192 SentencePiece tokens |
| Source / decoder limits | 1,024 / 512 |
| FFN | None |
| Positional encoding | RoPE on encoder and decoder self-attention; none on cross-attention |
| Other details | QK ZCRMSNorm, gated residuals, shared embeddings, tied output |

Artifact SHA-256:

```text
model.safetensors c5f9a3016e4537e492c362da5cb8ba05107d8595bec0d5ea5d8a65801db46531
needle.pkl        40a32e91d1d4197bf15ba559b74f6727c342dc8746918742fc7d8e2c1f18df40
tokenizer.model   0823f5b9133c68a8140addc5d7a425fa9119c4c8cb4a550363b4bffa4ba1c8c7
```

The PyTorch port was checked against the original JAX checkpoint on identical
inputs in BF16. The encoder mean absolute difference was `0.00814`; logits mean
absolute difference was `0.03938`; the first nine ranked token IDs matched.
This audit exposed and fixed one port bug: self-attention K/V must receive the
same normalized state as Q.

### PleIAs/SYNTH RAG

Dataset revision: `0d6813a2966662c39f22f0b9af28a0c1c9f7a437`.

| Metric | Exact value |
|---|---:|
| Parquet shards | 500 |
| Full compressed bytes | 236,335,243,511 |
| All rows | 77,908,583 |
| `exercise == "rag"` rows | 796,582 |
| RAG languages | English only |
| Dataset-reported RAG words | 595,812,155 |
| Remote metadata scan | 153.06 seconds |

RAG evidence is stored in `constraints` as numbered `<source_N>` blocks. The
answer often contains `<ref name="source_N">...</ref>` evidence citations, so
supporting source IDs can be recovered without using `synthetic_reasoning`.
The filtered RAG subset is being extracted directly from remote Parquet; the
236 GB full corpus is not downloaded.

### Compute and storage

The assigned host exposes one RTX 4090, BF16 support, 49,140 MiB VRAM, 20 vCPU,
and a 50 GB persistent volume at `/root/autodl-tmp`. All environments, caches,
datasets, runs, and checkpoints remain on that volume.

The faithful PyTorch port now uses scaled-dot-product attention. At the maximum
1,024-token source and 512-token decoder lengths:

| Mode | Batch | Padded tokens/s | Peak VRAM |
|---|---:|---:|---:|
| Eager BF16 | 12 | 242,347 | 5.30 GB |
| Compiled BF16 | 12 | 446,824 | 3.28 GB |
| Compiled BF16 | 24 | **452,397** | 6.49 GB |

The measured N1 training loop, including CE, z-loss, clipping, and AdamW,
sustained about 405k–412k padded tokens/s and 250k actual tokens/s after compile.

### Preliminary N1 data decision

A complete 1,590-row RAG shard showed that raw citation-heavy answers have a
median length of 623 tokens. Removing inline `<ref>quoted evidence</ref>` blocks
while retaining their source IDs for evidence selection reduces the median
target to 286 tokens. We therefore train N1 on the clean answer text and reserve
explicit provenance supervision for N2-PG.

The same shard had a median of five sources and only 15.5% of full contexts fit
1,024 tokens. Keeping all gold-cited sources first and filling the remainder
with deterministically shuffled distractors raised context fit to 39.5%.
Conservative unsupported-answer filtering plus the 512-token target limit left
399 usable rows (25.1%). This is a preliminary single-shard rate; the full
extracted corpus audit will replace it.

### Next measured gates

1. Freeze N0 QA/context-dependence evaluation sets.
2. Audit RAG source counts, citations, token lengths, and 1,024-token fit rate.
3. Benchmark N1 throughput and stable batch size.
4. Train N1 with query + evidence to raw answer CE and a small z-loss.
5. Fork the same N1 checkpoint into matched N2-GEN and N2-PG runs.
