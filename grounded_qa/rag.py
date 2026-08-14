from __future__ import annotations

import random
from typing import Any


def build_rag_packs(rows: list[dict[str, Any]], n: int, seed: int = 42, positive_ratio: float = 0.7) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    packs = []
    for index in range(n):
        positive = rng.random() < positive_ratio
        support = rng.choice(rows)
        distractors = rng.sample([row for row in rows if row is not support], k=min(rng.randint(3, 7), max(len(rows) - 1, 0)))
        chunks = [support["context"], *(row["context"] for row in distractors)]
        rng.shuffle(chunks)
        context = " ".join(f"[{chunk_index}] {chunk}" for chunk_index, chunk in enumerate(chunks, 1))
        if positive:
            packs.append({**support, "id": f"rag-positive-{index}", "context": context, "source": "rag", "phase": "rag"})
        else:
            packs.append({
                **support,
                "id": f"rag-negative-{index}",
                "context": context.replace(support["context"], "This chunk contains unrelated information."),
                "answer": "I don't know this.",
                "answerable": False,
                "evidence": "",
                "source": "rag",
                "phase": "rag",
            })
    return packs
