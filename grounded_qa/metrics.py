from __future__ import annotations

import re
from collections import Counter
from typing import Iterable


NUMBER_RE = re.compile(r"[$€£¥]?\d[\d,.]*(?:\s*(?:billion|million|thousand|%))?", re.I)
ENTITY_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z0-9-]+)+\b")


def normalize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def exact_match(prediction: str, gold: str) -> float:
    return float(" ".join(normalize(prediction)) == " ".join(normalize(gold)))


def token_f1(prediction: str, gold: str) -> float:
    pred, target = normalize(prediction), normalize(gold)
    if not pred or not target:
        return float(pred == target)
    overlap = sum((Counter(pred) & Counter(target)).values())
    if not overlap:
        return 0.0
    precision = overlap / len(pred)
    recall = overlap / len(target)
    return 2 * precision * recall / (precision + recall)


def _lcs_length(a: list[str], b: list[str]) -> int:
    previous = [0] * (len(b) + 1)
    for token_a in a:
        current = [0]
        for index, token_b in enumerate(b, 1):
            current.append(previous[index - 1] + 1 if token_a == token_b else max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def rouge_l(prediction: str, gold: str) -> float:
    pred, target = normalize(prediction), normalize(gold)
    if not pred or not target:
        return float(pred == target)
    lcs = _lcs_length(pred, target)
    precision = lcs / len(pred)
    recall = lcs / len(target)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def context_token_overlap(answer: str, context: str) -> float:
    answer_tokens = normalize(answer)
    context_tokens = set(normalize(context))
    if not answer_tokens:
        return 0.0
    return sum(token in context_tokens for token in answer_tokens) / len(answer_tokens)


def unsupported_number_rate(answer: str, context: str) -> float:
    numbers = NUMBER_RE.findall(answer)
    if not numbers:
        return 0.0
    context_lower = context.lower()
    return sum(number.lower() not in context_lower for number in numbers) / len(numbers)


def unsupported_entity_rate(answer: str, context: str, question: str) -> float:
    entities = ENTITY_RE.findall(answer)
    if not entities:
        return 0.0
    allowed = f"{context} {question}".lower()
    return sum(entity.lower() not in allowed for entity in entities) / len(entities)


def repeated_ngram_rate(texts: Iterable[str], n: int = 3) -> float:
    total = repeated = 0
    for text in texts:
        tokens = normalize(text)
        ngrams = [tuple(tokens[i : i + n]) for i in range(max(0, len(tokens) - n + 1))]
        total += len(ngrams)
        repeated += len(ngrams) - len(set(ngrams))
    return repeated / total if total else 0.0
