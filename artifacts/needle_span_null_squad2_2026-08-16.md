# Needle span + NULL SQuAD2 experiment

Date: 2026-08-16  
Code commit: `f18997b`  
W&B main run: [needle-span-null-squad2-8000](https://wandb.ai/tomartarun2001-adobe/grounded-attn-qa/runs/ndgdzilf)

## Question

Can the public pretrained Needle model learn both behaviours jointly from the
first downstream QA step?

```text
answerable paragraph   -> select the answer span
unanswerable paragraph -> select NULL
```

This experiment deliberately used no decoder generation, pointer-generator,
EOS refusal, verifier, NLI, adapter, LoRA, auxiliary loss, or score fusion.

## Starting model

- Public checkpoint: `Cactus-Compute/needle`, local file `model.safetensors`
- Model hash: `c5f9a3016e4537e492c362da5cb8ba05107d8595bec0d5ea5d8a65801db46531`
- Parameters: `26,233,372`
- Architecture: 12-layer encoder, 8-layer decoder, `d_model=512`, GQA 8 query / 4 KV heads, RoPE, ZCRMSNorm, no FFN
- Vocabulary: 8,192 SentencePiece tokens
- Tokenizer hash: `0823f5b9133c68a8140addc5d7a425fa9119c4c8cb4a550363b4bffa4ba1c8c7`
- Input format: query tokens, Needle tool separator token `5`, context tokens

The experiment used the encoder states. The decoder was loaded as part of the
public checkpoint but was not used by this head.

## Head and targets

The new head adds 2,052 parameters:

```text
logits[0]     = learned NULL candidate scored against the question summary
logits[i + 1] = score for source position i
```

Only context positions compete. Query and padding positions are masked.

```text
answerable:   gold start/end = exact context token positions + 1
unanswerable: gold start/end = 0, the NULL class
```

The loss was exactly:

```text
cross_entropy(start_logits, gold_start)
+ cross_entropy(end_logits, gold_end)
```

## Dataset

Dataset: `rajpurkar/squad_v2`, revision
`3ffb306f725f7d2ce8394bc1873b24868140c412`.

The tokenizer-aligned preparation used a 1,024-token source window. Answerable
answers were never silently truncated. Rows whose annotated answer could not
be aligned to complete tokenizer pieces were dropped and counted.

| split | official rows | usable | answerable | unanswerable | dropped |
|---|---:|---:|---:|---:|---:|
| train | 130,319 | 125,563 | 82,065 | 43,498 | 4,756 |
| validation | 11,873 | 11,635 | 5,690 | 5,945 | 238 |

All dropped rows were `token_boundary_alignment` failures. The prepared data
manifest hash is `2eb6b567f6129c19c898adcd7d63a160056a0005d214e067deec3dfea5b80770`.

## Mechanical gates

Each gate started again from the untouched public checkpoint. The higher head
learning rate was selected after the first 500-step answerable gate was still
learning slowly. The final gate settings were:

```text
AdamW
backbone LR = 2e-5
head LR     = 5e-4
batch size  = 8
precision   = bfloat16
steps       = 2,000
```

Results:

| gate | result |
|---|---|
| 64 answerable rows | 98.4% start, 100% end, 97.9% span F1 |
| 64 unanswerable rows | 100% NULL start and 100% NULL end |
| 32 answerable + 32 unanswerable | 100% answerable start/end, 100% NULL start/end, 100% thresholded EM/F1 |

These gates show that the new head and NULL competition are mechanically
learnable. The full-data result is therefore not a NULL plumbing failure.

## Full run

```text
steps:       8,000
batch size:  32
backbone LR: 2e-5
head LR:     5e-4
precision:   bfloat16
optimizer:   AdamW
GPU:         RTX 4090, peak observed about 8.6 GB VRAM
```

The run completed normally and saved step checkpoints from 1,000 through
8,000. No NaN, OOM, or runaway gradient occurred.

Validation trend:

| step | HasAns start | HasAns end | NoAns start | NoAns end | threshold F1 |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.18% | 0.53% | 0.00% | 0.00% | 51.12% |
| 500 | 0.00% | 0.02% | 99.85% | 99.98% | 51.12% |
| 2,000 | 6.49% | 5.92% | 87.89% | 87.35% | 51.13% |
| 4,000 | 11.21% | 11.53% | 83.57% | 80.66% | 51.32% |
| 8,000 | 13.73% | 14.36% | 83.50% | 79.92% | 51.76% |

## Final evaluation

The best threshold was selected by a SQuAD2-style sweep of:

```text
NULL score - best non-NULL span score
```

Best threshold: `-1.5625`.

| metric | result |
|---|---:|
| overall EM | 51.50% |
| overall F1 | 51.76% |
| HasAns EM | 4.01% |
| HasAns F1 | 4.53% |
| NoAns accuracy | 96.96% |
| false-refusal rate | 94.01% |
| false-answer rate | 3.04% |

If we ignore NULL and always take the best non-NULL span:

| metric | result |
|---|---:|
| HasAns EM | 29.49% |
| HasAns F1 | 44.77% |
| NoAns false-answer rate | 100% |

This separates the two problems. The model has some span extraction ability,
but its NULL-vs-span calibration is too conservative: it rejects almost every
question, including answerable ones.

## Qualitative examples

### Correct answer

```text
Question: What was the name of the Norman castle?
Context: ... They even lent their ethnicity to the name of their castle:
         Afranji, meaning "Franks." ...
Gold:       Afranji
Prediction: Afranji
```

### Correct refusal

```text
Question: Who gave their name to Normandy in the 1000's and 1100's?
Context: The Normans ... gave their name to Normandy, a region in France.
Gold:       no answer
Prediction: NULL
```

### False refusal

```text
Question: In what country is Normandy located?
Context: ... Normandy, a region in France. ...
Gold:       France
Prediction: NULL
```

### False answer

```text
Question: Who's aristocracy eventually served as avid Crusaders?
Context: ... Many Normans of Italy, France and England eventually served as
         avid Crusaders ...
Gold:       no answer
Prediction: Many Normans of Italy, France and England
```

## Conclusion

The clean joint formulation **can learn both behaviours on tiny fixed data**.
On the full natural SQuAD2 distribution, this two-pass run did not reach a
useful QA model. It learned a strong NULL bias and only weak real span
extraction, producing high NoAns accuracy mainly by refusing too much.

Therefore this experiment does **not** support the claim that pretrained
attention-only Needle cannot learn answerability. It supports the narrower
claim that this first full-data fine-tuning setup, with this optimizer and
8,000-step budget, did not transfer the tiny-gate behaviour to natural SQuAD2
well enough.

## Primary references

- [Needle model card](https://huggingface.co/Cactus-Compute/needle)
- [Google BERT SQuAD2 implementation](https://github.com/google-research/bert)
- [SQuAD2 dataset and official evaluation](https://rajpurkar.github.io/SQuAD-explorer/)
