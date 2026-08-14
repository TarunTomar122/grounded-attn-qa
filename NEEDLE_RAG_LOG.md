# Needle 26M grounded RAG log

## 2026-08-14 — starting-point audit

The available public checkpoint is the function-call post-trained Needle model,
not a verified clean foundation checkpoint. Its untouched free-running output
on five QA probes (correct, wrong, empty, and counterfactual contexts) was always:

```text
<tool_call> []</s>
```

This makes restoration of contextual question answering the purpose of N1.

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

### Next measured gates

1. Freeze N0 QA/context-dependence evaluation sets.
2. Audit RAG source counts, citations, token lengths, and 1,024-token fit rate.
3. Benchmark N1 throughput and stable batch size.
4. Train N1 with query + evidence to raw answer CE and a small z-loss.
5. Fork the same N1 checkpoint into matched N2-GEN and N2-PG runs.
