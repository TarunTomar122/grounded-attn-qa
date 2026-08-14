from __future__ import annotations

import random
import re
from typing import Any


REFUSAL = "I don't know this."


def _negative(row: dict[str, Any], context: str, kind: str, index: int) -> dict[str, Any]:
    return {
        **row,
        "id": f"{row['id']}-negative-{kind}-{index}",
        "context": context,
        "answer": REFUSAL,
        "answerable": False,
        "evidence": "",
        "source": kind,
        "phase": "refuse",
        "hardness": "hard" if kind in {"squad2_no", "conflict"} else "medium",
    }


def cross_document_negatives(rows: list[dict[str, Any]], n: int, seed: int = 42) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    result = []
    for index in range(n):
        question_row = rng.choice(rows)
        context_row = rng.choice(rows)
        tries = 0
        while question_row["context"] == context_row["context"] and tries < 20:
            context_row = rng.choice(rows)
            tries += 1
        if question_row["answer"].lower() in context_row["context"].lower():
            continue
        result.append(_negative(question_row, context_row["context"], "cross_document", index))
    return result


def numeric_date_negatives(rows: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    result = []
    for index, row in enumerate(rows[:n]):
        question = re.sub(r"\b(19|20)\d{2}\b", lambda match: str(int(match.group()) + 1), row["question"])
        if question == row["question"]:
            question = f"What was the value in a later year? {row['question']}"
        altered = {**row, "question": question}
        result.append(_negative(altered, row["context"], "numeric_date", index))
    return result


def entity_swap_negatives(rows: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    result = []
    for index, row in enumerate(rows[:n]):
        question = row["question"] + " involving the other person"
        result.append(_negative({**row, "question": question}, row["context"], "entity_swap", index))
    return result


def conflicting_negatives(rows: list[dict[str, Any]], n: int, seed: int = 42) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    result = []
    for index in range(n):
        first = rng.choice(rows)
        second = rng.choice(rows)
        if first["context"] == second["context"]:
            continue
        context = f"[1] {first['context']} [2] {second['context']}"
        result.append(_negative(first, context, "conflict", index))
    return result
