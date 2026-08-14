from __future__ import annotations

import re
import random
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def _row(
    *,
    row_id: str,
    question: str,
    context: str,
    answer: str,
    evidence: str,
    source: str,
    hardness: str,
    metadata: dict[str, Any] | None = None,
    answer_start: int | None = None,
    answer_end: int | None = None,
) -> dict[str, Any]:
    row = {
        "id": row_id,
        "question": question.strip(),
        "context": context,
        "answer": answer,
        "answerable": bool(answer),
        "evidence": evidence,
        "source": source,
        "phase": "real_copy",
        "hardness": hardness,
        "metadata": metadata or {},
    }
    if answer_start is not None and answer_end is not None:
        row["answer_start"] = answer_start
        row["answer_end"] = answer_end
    return row


def _question_type(question: str) -> str:
    words = question.lower().lstrip().split()
    first = words[0] if words else ""
    if first in {"who", "what", "when", "where", "which", "why"}:
        return first
    if first == "how":
        return "how_many" if any(word in {"many", "much"} for word in words[1:3]) else "how"
    return "other"


def _lexical_overlap(question: str, sentence: str) -> str:
    question_words = set(re.findall(r"[a-z]+", question.lower()))
    sentence_words = set(re.findall(r"[a-z]+", sentence.lower()))
    ratio = len(question_words & sentence_words) / max(len(question_words), 1)
    return "high" if ratio >= 0.45 else "medium" if ratio >= 0.2 else "low"


def squad2_rows(split: str = "train", max_n: int | None = None) -> list[dict[str, Any]]:
    rows, _ = squad2_rows_with_stats(split=split, max_n=max_n)
    return rows


def squad2_rows_with_stats(
    *,
    split: str,
    tokenizer=None,
    source_length: int = 512,
    target_length: int = 64,
    max_n: int | None = None,
    seed: int = 42,
    validation: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    from datasets import load_dataset

    try:
        dataset = load_dataset("rajpurkar/squad_v2", split=split)
    except ValueError as error:
        if "Feature type 'List' not found" not in str(error):
            raise
        dataset = _read_cached_squad2_arrow(split)
    return squad2_rows_from_items(
        dataset,
        split=split,
        tokenizer=tokenizer,
        source_length=source_length,
        target_length=target_length,
        max_n=max_n,
        seed=seed,
        validation=validation,
    )


def _read_cached_squad2_arrow(split: str) -> list[dict[str, Any]]:
    """Read old cached SQuAD Arrow files when datasets 3.6 cannot decode List metadata."""
    import pyarrow as pa
    from datasets import config

    candidates = sorted(
        Path(config.HF_DATASETS_CACHE).glob(
            f"rajpurkar___squad_v2/**/squad_v2-{split}.arrow"
        )
    )
    if not candidates:
        raise FileNotFoundError(f"cached SQuAD2 Arrow split not found: {split}")
    with pa.memory_map(str(candidates[-1]), "r") as source:
        try:
            table = pa.ipc.open_stream(source).read_all()
        except (pa.ArrowInvalid, pa.ArrowIOError):
            source.seek(0)
            table = pa.ipc.open_file(source).read_all()
    return table.to_pylist()


def squad2_rows_from_items(
    items: Iterable[dict[str, Any]],
    *,
    split: str,
    tokenizer,
    source_length: int = 512,
    target_length: int = 64,
    max_n: int | None = None,
    seed: int = 42,
    validation: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Prepare answerable SQuAD2 rows with strict source-span alignment."""
    stats = {
        "seen": 0,
        "unanswerable": 0,
        "empty_answer": 0,
        "invalid_span": 0,
        "alignment_dropped": 0,
        "kept": 0,
        "windowed": 0,
    }
    rows = []
    for index, item in enumerate(items):
        stats["seen"] += 1
        answers = item.get("answers", {})
        pairs = [
            (text, int(start))
            for text, start in zip(answers.get("text", []), answers.get("answer_start", []))
            if text
        ]
        if not pairs:
            stats["unanswerable"] += 1
            continue
        answer, start = pairs[0]
        context = item["context"]
        end = start + len(answer)
        if start < 0 or context[start:end] != answer:
            stats["invalid_span"] += 1
            continue
        prepared = _prepare_squad2_item(
            item,
            index=index,
            split=split,
            tokenizer=tokenizer,
            source_length=source_length,
            target_length=target_length,
            seed=seed,
            validation=validation,
        )
        if prepared is None:
            stats["alignment_dropped"] += 1
            continue
        row, windowed = prepared
        if windowed:
            stats["windowed"] += 1
        rows.append(row)
        stats["kept"] += 1
        if max_n is not None and len(rows) >= max_n:
            break
    return rows, stats


def _prepare_squad2_item(
    item: dict[str, Any],
    *,
    index: int,
    split: str,
    tokenizer,
    source_length: int,
    target_length: int,
    seed: int,
    validation: bool,
) -> tuple[dict[str, Any], bool] | None:
    answers = item["answers"]
    answer = next(text for text in answers["text"] if text)
    answer_index = next(i for i, text in enumerate(answers["text"]) if text)
    start = int(answers["answer_start"][answer_index])
    end = start + len(answer)
    context = item["context"]
    if tokenizer is None:
        return _row(
            row_id=f"squad2-{split}-{item.get('id', index)}",
            question=item["question"],
            context=context,
            answer=answer,
            evidence=answer,
            source="squad2",
            hardness="medium",
            answer_start=start,
            answer_end=end,
            metadata={"title": item.get("title", ""), "answer_start": start, "answer_end": end},
        ), False
    encoded = tokenizer.encode(context, add_special_tokens=False)
    answer_positions = _span_token_positions(encoded.offsets, start, end)
    answer_ids = [encoded.ids[position] for position in answer_positions]
    if not answer_positions or tokenizer.decode(answer_ids).strip() != answer.strip():
        return None
    question_length = len(tokenizer.encode(item["question"], add_special_tokens=False).ids)
    context_budget = source_length - question_length - 5
    if len(answer_positions) > context_budget:
        return None
    if len(answer_positions) > target_length - 1:
        return None

    windowed = len(encoded.ids) > context_budget
    if windowed:
        extra = context_budget - len(answer_positions)
        left_min = max(0, answer_positions[-1] + 1 + extra - len(encoded.ids))
        left_max = min(answer_positions[0], extra)
        if validation:
            left = left_max
        else:
            left = random.Random(seed + index).randint(left_min, left_max)
        window_start = answer_positions[0] - left
        window_end = answer_positions[-1] + 1 + extra - left
        source_start = encoded.offsets[window_start][0]
        source_end = encoded.offsets[window_end - 1][1]
        context_start = source_start
        context_end = source_end
        window_context = context[context_start:context_end]
        window_answer_start = start - context_start
        window_answer_end = end - context_start
        positions = [position - window_start for position in answer_positions]
    else:
        context_start = 0
        window_context = context
        window_answer_start = start
        window_answer_end = end
        positions = answer_positions
    if window_context[window_answer_start:window_answer_end] != answer:
        return None
    sentence_start = max(context.rfind(".", 0, start), context.rfind("!", 0, start), context.rfind("?", 0, start)) + 1
    sentence_end_candidates = [position for mark in ".!?" if (position := context.find(mark, end)) >= 0]
    sentence_end = min(sentence_end_candidates, default=len(context))
    answer_sentence = context[sentence_start:sentence_end].strip()
    context_budget = source_length - question_length - 5
    while len(tokenizer.encode(window_context, add_special_tokens=False).ids) > context_budget:
        if window_answer_start > 0:
            boundary = window_context.find(" ")
            trim = boundary + 1 if 0 <= boundary < window_answer_start else 1
            window_context = window_context[trim:]
            context_start += trim
            window_answer_start -= trim
            window_answer_end -= trim
        elif window_answer_end < len(window_context):
            window_context = window_context[:window_answer_end]
        else:
            return None
    if window_context[window_answer_start:window_answer_end] != answer:
        return None
    row = _row(
        row_id=f"squad2-{split}-{item.get('id', index)}",
        question=item["question"],
        context=window_context,
        answer=answer,
        evidence=answer,
        source="squad2",
        hardness="medium",
        answer_start=start,
        answer_end=end,
        metadata={
            "title": item.get("title", ""),
            "relation": "squad2",
            "template_family": "squad2",
            "question_structure": "human_question",
            "context_structure": "windowed" if windowed else "full",
            "question_type": _question_type(item["question"]),
            "lexical_overlap_bucket": _lexical_overlap(item["question"], answer_sentence),
            "answer_start": start,
            "answer_end": end,
            "window_answer_start": window_answer_start,
            "window_answer_end": window_answer_end,
            "window_context_start": context_start,
            "gold_context_token_start": positions[0],
            "gold_context_token_end": positions[-1] + 1,
            "accepted_answer_count": len(answers["text"]),
            "windowed": windowed,
            "tokenizer_boundary": "raw",
        },
    )
    return row, windowed


def _span_token_positions(offsets: list[tuple[int, int]], start: int, end: int) -> list[int]:
    """Map original context chars into the byte-level source token span."""
    source_start, source_end = start, end
    return [
        index
        for index, (token_start, token_end) in enumerate(offsets)
        if token_end > source_start and token_start < source_end
    ]


def natural_questions_rows(split: str = "train", max_n: int | None = None) -> list[dict[str, Any]]:
    """Load short-answer NQ rows and window context around the annotated answer."""
    from datasets import load_dataset

    dataset = load_dataset("google-research-datasets/natural_questions", split=split)
    rows = []
    for index, item in enumerate(dataset):
        annotations = item.get("annotations", [])
        if not annotations:
            continue
        annotation = annotations[0]
        short_answers = annotation.get("short_answers", [])
        if not short_answers:
            continue
        document = item.get("document", {})
        tokens = document.get("tokens", {})
        words = tokens.get("token", [])
        answer = " ".join(words[short_answers[0]["start_token"] : short_answers[0]["end_token"]]).strip()
        if not answer or len(answer.split()) > 12:
            continue
        context = " ".join(words)
        rows.append(
            _row(
                row_id=f"nq-{split}-{item.get('id', index)}",
                question=item.get("question", {}).get("text", item.get("question_text", "")),
                context=context,
                answer=answer,
                evidence=answer,
                source="natural_questions",
                hardness="medium",
                metadata={"document_id": item.get("document_id", "")},
            )
        )
        if max_n is not None and len(rows) >= max_n:
            break
    return rows


def coqa_rows(split: str = "train", max_n: int | None = None) -> list[dict[str, Any]]:
    from datasets import load_dataset

    dataset = load_dataset("stanfordnlp/coqa", split=split)
    rows = []
    for item in dataset:
        story = item.get("story", "")
        questions = item.get("questions", [])
        answers = item.get("answers", {})
        texts = answers.get("input_text", [])
        starts = answers.get("answer_start", [])
        ends = answers.get("answer_end", [])
        history: list[str] = []
        for index, question in enumerate(questions):
            original_answer = texts[index].strip() if index < len(texts) else ""
            if not original_answer:
                continue
            answer = "" if original_answer.lower() == "unknown" else original_answer
            start = starts[index] if index < len(starts) else -1
            end = ends[index] if index < len(ends) else -1
            evidence = story[start:end] if 0 <= start < end <= len(story) else ""
            useful_history = " ".join(history[-4:])
            current = f"Previous turns: {useful_history} Current question: {question}" if useful_history else question
            rows.append(
                _row(
                    row_id=f"coqa-{item.get('id', len(rows))}-{index}",
                    question=current,
                    context=story,
                    answer=answer,
                    evidence=evidence,
                    source="coqa",
                    hardness="medium",
                    metadata={"conversation_id": item.get("id", ""), "answer_start": start, "answer_end": end, "original_answer": original_answer},
                )
            )
            history.extend((f"Q: {question}", f"A: {original_answer}"))
            if max_n is not None and len(rows) >= max_n:
                return rows
    return rows


def msmarco_rows(split: str = "train", max_n: int | None = None) -> list[dict[str, Any]]:
    from datasets import load_dataset

    dataset = load_dataset("microsoft/ms_marco", "v1.1", split=split)
    rows = []
    for index, item in enumerate(dataset):
        answers = item.get("answers", [])
        answer = answers[0].strip() if answers else ""
        passages = item.get("passages", {})
        texts = passages.get("passage_text", [])
        selected = passages.get("is_selected", [])
        context = " ".join(text for text, flag in zip(texts, selected) if flag).strip()
        if not answer or not context or len(answer.split()) > 64 or not _numbers_supported(answer, context):
            continue
        rows.append(
            _row(
                row_id=f"msmarco-{split}-{item.get('query_id', index)}",
                question=item.get("query", ""),
                context=context,
                answer=answer,
                evidence=context,
                source="msmarco",
                hardness="hard",
                metadata={"query_id": item.get("query_id", "")},
            )
        )
        if max_n is not None and len(rows) >= max_n:
            break
    return rows


def _numbers_supported(answer: str, context: str) -> bool:
    numbers = re.findall(r"[$€£¥]?\d[\d,.]*", answer)
    context_lower = context.lower()
    return all(number.lower() in context_lower for number in numbers)


def deduplicate(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    result = []
    for row in rows:
        key = (" ".join(row["question"].lower().split()), " ".join(row["context"].lower().split()))
        if key in seen:
            continue
        seen.add(key)
        result.append(row)
    return result
