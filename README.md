# Grounded attention-only reader

A ~24M **attention-only** decoder (no MLP) that answers a question **only** if the answer is in the prompt. Otherwise it emits `I don't know this.`

Meant as a hallucination-free last mile for RAG: retriever stuffs chunks in the window, this model binds Q → span or refuses. No parametric world knowledge.

Full writeup: [GOAL.md](GOAL.md)

## Status (2026-08-13)

Trained on Colab T4.

| run | result |
|---|---|
| 3k steps, SQuAD 2.0 mix | refuse-collapse: always `I don't know this.` |
| 4k steps, **copy-only** curriculum | **EM 0.375** — synth facts copy correctly (`$2.1 billion`); SQuAD spans still fail |

Phase A gate (copy works) is almost hit. Phase B (mix refuse) is next, synth-only first.

## Train (Colab T4)

```bash
# this VPS cannot train it (no GPU). Colab CLI default timeout is 30s — set it.
colab run --gpu T4 --timeout 3600 train.py --phase copy --steps 4000 --train-n 20000
```

Phases:

- `copy` — 100% answerable (synth templates + SQuAD yes). Teach copy first.
- `mix` — ~70% yes / 30% refuse (synth traps + cross-paired SQuAD). Only after copy works.

## Contract

```
Context: <retrieved text>

Question: <user question>

Answer:
```

Output is either a span that occurs in Context, or `I don't know this.`

## Layout

- `train.py` — model + Colab-ready train/eval loop (self-contained)
- `data.py` — curriculum builders (also inlined in `train.py` for `colab run`)
- `GOAL.md` — metrics, kill criteria, RAG deployment
