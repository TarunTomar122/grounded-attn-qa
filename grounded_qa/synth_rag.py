from __future__ import annotations

import hashlib
import re

from .needle_tokenizer import NeedleTokenizer


SOURCE_RE = re.compile(r"<source_(\d+)>\s*(.*?)\s*</source_\1>", re.DOTALL)
REF_RE = re.compile(r'<ref\s+name="source_(\d+)">.*?</ref>', re.DOTALL)
UNSUPPORTED_RE = re.compile(
    r"\b(?:sources?|documents?|texts?|provided (?:source|sources|information|texts?))\b.{0,100}"
    r"\b(?:do(?:es)?\s+not|cannot|fail(?:s)?\s+to|insufficient|not\s+sufficient)\b"
    r"|\b(?:there\s+is\s+)?(?:not|no)\s+(?:enough\s+|sufficient\s+)?information\b"
    r"|\bit\s+is\s+not\s+possible\s+to\b"
    r"|\b(?:this|the)\s+question\s+cannot\s+be\s+answered\b"
    r"|\bi(?:\s+am\s+sorry,?\s+but\s+i)?\s+cannot\s+answer\b",
    re.IGNORECASE | re.DOTALL,
)


def parse_sources(constraints: str) -> list[tuple[str, str]]:
    return [(source_id, text.strip()) for source_id, text in SOURCE_RE.findall(constraints)]


def cited_source_ids(answer: str) -> list[str]:
    return list(dict.fromkeys(REF_RE.findall(answer)))


def clean_answer(answer: str) -> str:
    return re.sub(r"\s+", " ", REF_RE.sub("", answer)).strip()


def appears_unsupported(answer: str) -> bool:
    return bool(UNSUPPORTED_RE.search(clean_answer(answer)[:500]))


def evidence_context(
    *,
    row_id: str,
    query: str,
    constraints: str,
    answer: str,
    tokenizer: NeedleTokenizer,
    source_length: int = 1024,
) -> str | None:
    """Keep all cited sources, then fill the remaining budget with distractors."""
    sources = parse_sources(constraints)
    by_id = dict(sources)
    cited = cited_source_ids(answer)
    if not cited or any(source_id not in by_id for source_id in cited):
        return None
    budget = source_length - len(tokenizer.encode(query)) - 1
    if budget <= 0:
        return None

    def tagged(source_id: str) -> str:
        return f"<source_{source_id}> {by_id[source_id]} </source_{source_id}>"

    selected = [tagged(source_id) for source_id in cited]
    context = "\n\n".join(selected)
    if len(tokenizer.encode(context)) > budget:
        return None
    distractors = [source_id for source_id, _ in sources if source_id not in cited]
    distractors.sort(key=lambda source_id: hashlib.sha256(f"{row_id}:{source_id}".encode()).digest())
    for source_id in distractors:
        candidate = "\n\n".join([*selected, tagged(source_id)])
        if len(tokenizer.encode(candidate)) <= budget:
            selected.append(tagged(source_id))
    return "\n\n".join(selected)
