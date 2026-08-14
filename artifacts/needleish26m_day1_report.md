# Needleish26M Foundation Pretraining — Day 1

## Result

The run ended at `150003051` total source+target token exposures. Estimated training wall time was `6715` seconds from the steady-state training log; the exact-checkpoint evaluation took `72.0944244125858` seconds. The report is evidence-only; no downstream SQuAD, pointer, refusal, SFT, or RL training was run.

## Architecture

```json
{
  "parameters": 26235420,
  "model_name": "needleish26m_v1",
  "vocab_size": 8196,
  "d_model": 512,
  "encoder_layers": 12,
  "decoder_layers": 8,
  "query_heads": 8,
  "kv_heads": 4,
  "head_dim": 64,
  "rope_theta": 10000.0,
  "rmsnorm_eps": 1e-06,
  "source_length": 512,
  "target_length": 256,
  "pad_id": 0,
  "eos_id": 1,
  "bos_id": 2,
  "unk_id": 3,
  "query_id": 8192,
  "context_id": 8193,
  "reasoning_id": 8194,
  "answer_id": 8195,
  "dropout": 0.0
}
```

Parameter breakdown: `{}`.

## Data

- Dataset: PleIAs/SYNTH, English-only after filtering.
- Shards: `[['synth_001.parquet'], ['synth_002.parquet'], ['synth_003.parquet'], ['synth_004.parquet']]`.
- Train/validation rows: `6207` / `136`.
- Unique source URLs: `5056` / `90`.
- Global source URL intersection: `0`.
- Train exercise distribution after the deterministic cap: `{'math mcq': 2612, 'mcq': 3000, 'memorization': 544, 'rag': 44, 'cooking': 7}`.
- Validation answer-overlap buckets: `{'medium': 87, 'high': 37, 'low': 12}`.
- Source/target limits: 512 / 256 tokens.

## Learning and grounding

- Final metrics: `{"answer_token_accuracy": 0.839975845410628, "context_dependency_gap": 2.652970818912282, "elapsed_seconds": 72.0944244125858, "evaluation_elapsed_seconds": 72.0944244125858, "exercise/math mcq/answer_loss": 0.8574893136586373, "exercise/math mcq/answer_token_accuracy": 0.84375, "exercise/math mcq/loss": 3.1978828821863448, "exercise/mcq/answer_loss": 0.6138495511734633, "exercise/mcq/answer_token_accuracy": 0.8987052551408987, "exercise/mcq/loss": 2.874002932333479, "exercise/memorization/answer_loss": 3.593715486978355, "exercise/memorization/answer_token_accuracy": 0.5586034912718204, "exercise/memorization/loss": 4.060134792327881, "exercise/rag/answer_loss": 5.548778903099798, "exercise/rag/answer_token_accuracy": 0.3870967741935484, "exercise/rag/loss": 5.5482048988342285, "loss_answer": 1.1273319295063111, "loss_reasoning": 3.201243557985692, "loss_total": 3.0039551117840935, "overlap/high/answer_loss": 0.8661425374842093, "overlap/high/answer_token_accuracy": 0.8625792811839323, "overlap/high/loss": 2.9952282067891716, "overlap/low/answer_loss": 0.8780362116608847, "overlap/low/answer_token_accuracy": 0.8865248226950354, "overlap/low/loss": 2.696620066960653, "overlap/medium/answer_loss": 1.2541794478949788, "overlap/medium/answer_token_accuracy": 0.8274157303370786, "overlap/medium/loss": 3.050057634539988, "reasoning_token_accuracy": 0.5909725749111224, "rows_seen": 263967, "source_tokens_seen": 82437549, "step": 33001, "target_tokens_seen": 67565502, "tokens_seen": 150003051, "training_elapsed_seconds_estimate": 6714.518618915434, "unique_rows_seen": 6206}`.
- Training curve points: `241`.
- Validation curve points: `66`.
- Qualitative probe checkpoints: `9`.
- Peak training reserved GPU memory: `5.77 GB`.
- Measured steady-state training throughput: `22341` source+target tokens/sec.
- M2 Max benchmark: `benchmarks/MPS_2026-08-13.md`.
- Context-dependency gap is recorded in each validation point as `val/context_dependency_gap`.

## Throughput estimates

At the measured throughput, estimated wall-clock time is `{"0.5B": 6.21682964527211, "1B": 12.43365929054422, "5B": 62.1682964527211}` hours for 500M, 1B, and 5B tokens.

## Recommendation

Use the curves above to decide whether to scale the same foundation run tomorrow. Keep downstream specialization out of this decision.
