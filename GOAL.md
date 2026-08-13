# Grounded attention-only reader

## One sentence

A ~24M **attention-only** decoder that answers a question **only** if the answer is in the prompt (retrieved docs / tables / notes). If it is not there, it always emits `I don't know this.` It must never use parametric “world knowledge.”

## Why this is the experiment

Needle / SAN: attention is a **router over the window**, not a fact store. FFNs are the warehouse. RAG wants the opposite of a warehouse — the truth is in the retrieved text. A no-FFN decoder that can only bind Q → span in context is the right inductive bias for a hallucination-free RAG cap.

End use: any retriever (garden, papers, 10-Ks, chat files) stuffs chunks into the window. This model is the last mile. Same weights, any domain.

## Contract

Input:

```
Context: <retrieved text>

Question: <user question>

Answer:
```

Output is **exactly one** of:

1. A span (or short extractive phrase) **that occurs in Context**
2. The fixed string `I don't know this.`

Illegal:

- Answering from pretraining memory
- Paraphrasing a fact that is not in Context
- Refusing when the span is clearly present
- Inventing numbers, names, citations

## What already happened (2026-08-13)

| run | steps | result |
|---|---|---|
| smoke 200 | empty / EOS | packing bug (first answer token untrained) |
| 3k T4 | **refuse_recall = 1.0**, **EM = 0**, **false_refuse = 1.0** | refuse-collapse |

The model found the cheap policy: always refuse. That string is constant and easy. Copying a variable span is hard. SQuAD 2.0 paper ([Rajpurkar et al. 2018](https://arxiv.org/abs/1806.03822)): a system that always abstains already gets ~49 F1; existing models sat closer to that baseline than to humans. We reproduced the same local minimum at 24M from scratch.

## Why it collapsed

1. **Asymmetric difficulty.** Refuse = one sequence. Answer = thousands of spans.
2. **No copy curriculum.** Random-init attention-only never learned “point at the context” before we mixed negatives.
3. **SQuAD 2.0 negatives are adversarial** (entity swap, antonym, plausible distractors). Too hard as the *first* refuse signal.
4. **3k steps / 16k rows** is not enough for a 24M LM from scratch to induce copying.
5. **Generative EM is harsh.** Span-pointer heads (BiDAF / DocQA) are the usual SQuAD setup; we generate tokens. Copy has to be *learned*, not built in.

## How we make it work

### Phase A — teach copy (no refuse)

100% answerable.

- Synthetic templates: fact in sentence → ask that fact. Answer is a **verbatim substring**.
- SQuAD 1.1 / SQuAD 2.0 **answerable only**.
- Easy lookups: who / when / how much, answer length ≤ 8 tokens.
- Success gate: **EM ≥ 0.40** on held-out answerable before Phase B. If we can’t copy, we do not add refuse.

### Phase B — teach abstain (easy negatives first)

Keep ~70% answerable / 30% refuse.

Negatives in order of hardness (SQuAD 2.0 paper: auto negatives are easier than crowd adversarial):

1. **Cross-pair:** question from doc A + context from doc B (no overlap).
2. **Entity swap:** same paragraph, question names an entity that is not there.
3. **Official SQuAD 2.0 unanswerable** last.

Loss: answerable examples upweighted (or refuse downweighted) so the constant string cannot dominate the gradient.

### Phase C — RAG-shaped packs

Multiple chunks in one context (`[1] ... [2] ...`). Question only supported by one chunk. Also packs where **no** chunk supports it → refuse. This is the deployment format.

### Optional later

- Finance slice (filings / prices) as a *transfer* test, not the train set.
- Pointer / constrained decode: only emit tokens that appear in context, or REFUSE. Grammar like Needle.
- Special `<refuse>` token instead of a phrase (shorter, cleaner decision).

## Dataset we are building

`experiments/grounded_qa/data.py` emits rows:

```
{context, question, answer, answerable, source, hardness}
```

Sources:

| source | answerable | role |
|---|---|---|
| `synth_copy` | yes | force verbatim copy |
| `synth_trap` | no | same style, fact missing |
| `squad_yes` | yes | natural extractive QA |
| `squad_cross` | no | easy refuse |
| `squad_official_no` | no | hard refuse (phase B+) |

Held-out val: mix of `synth_*` and SQuAD val, never used in train.

## Metrics (kill criteria)

On a fixed val mix (half yes / half no):

| metric | meaning | ship | kill this approach |
|---|---|---|---|
| **EM** | exact / containment on answerable | ≥ 0.35 then ≥ 0.50 | still 0 after Phase A |
| **refuse_recall** | refuse when no | ≥ 0.85 | < 0.5 after Phase B |
| **false_refuse** | refuse when yes | ≤ 0.25 then ≤ 0.15 | stays ~1.0 |
| **hallucination** | answerable=no but emitted a non-refuse string **not** in context | ≈ 0 | > 5% |

Primary product metric: **false answers** (answered when it should refuse). That must stay near zero even if EM is mediocre.

## Model

Stay attention-only (no MLP), ~24–45M, GPT-2 tokenizer, block 512.

```
n_layer=8, n_embd=384, n_head=8  →  24.2M   (current)
n_layer=12, n_embd=512, n_head=8 →  ~38M    (if copy saturates)
```

Train on Colab T4 (`colab run --gpu T4 --timeout 7200`). This VPS has no GPU.

## RAG deployment (after metrics pass)

```
retrieve(q) → top-k chunks
pack into Context
model → answer | refuse
if refuse and k was small: retrieve more / widen / give up
```

The retriever can be anything (current garden RAG, BM25, embeddings). This repo owns the **reader**.

## Out of scope

- Open-ended chat
- Multi-hop synthesis / related-work writing
- Beating 7B generalists on TriviaQA without retrieval
- Cloning Needle device-tool calling

## Status

- [x] Attention-only decoder + Colab loop
- [x] Reproduce refuse-collapse
- [x] Write this goal
- [x] Phase A dataset + train — **EM 0.375** after 4k steps (synth copy works; SQuAD still fails)
- [ ] Phase B mix-in refuse
- [ ] Phase C multi-chunk RAG packs
- [ ] Plug into a real retriever and dogfood
