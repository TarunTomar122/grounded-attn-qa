from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .needle_tokenizer import EOS_ID, TOOLS_ID


SQUAD2_DATASET = "rajpurkar/squad_v2"
SQUAD2_REVISION = "3ffb306f725f7d2ce8394bc1873b24868140c412"
COQA_DATASET = "stanfordnlp/coqa"
COQA_REVISION = "0d9e9952f1ef6e5415492d3d84b5873259137e3c"


@dataclass(frozen=True)
class Piece:
    id: int
    begin: int
    end: int


@dataclass
class PreparedNeedleQA:
    source_ids: list[int]
    target_ids: list[int]
    gold_copy_positions: list[int]
    context_start: int
    example_index: int
    turn_index: int
    evidence_start: int
    evidence_end: int
    window_start: int


def immutable_pieces(tokenizer, text: str) -> list[Piece]:
    """Return SentencePiece IDs with its UTF-8 byte offsets converted to characters."""
    proto = tokenizer.sp.encode(text, return_type="proto")
    byte_to_character = {0: 0}
    byte_offset = 0
    for character_index, character in enumerate(text, 1):
        byte_offset += len(character.encode("utf-8"))
        byte_to_character[byte_offset] = character_index
    return [
        Piece(
            int(piece.id),
            byte_to_character[int(piece.begin)],
            byte_to_character[int(piece.end)],
        )
        for piece in proto.pieces
    ]


def span_piece_indices(pieces: list[Piece], start: int, end: int) -> list[int]:
    """Select pieces touching [start, end), including zero-width byte-fallback pieces."""
    return [
        index
        for index, piece in enumerate(pieces)
        if (piece.end > start and piece.begin < end)
        or (piece.begin == piece.end and start <= piece.begin < end)
    ]


def _covers_only_span_or_whitespace(text: str, pieces: list[Piece], positions: list[int], start: int, end: int) -> bool:
    covered_start = pieces[positions[0]].begin
    covered_end = pieces[positions[-1]].end
    prefix = text[covered_start:start]
    suffix = text[end:covered_end]
    return (
        covered_start <= start
        and covered_end >= end
        and (not prefix or prefix.isspace())
        and (not suffix or suffix.isspace())
    )


def _evidence_window(
    tokenizer,
    query: str,
    context: str,
    evidence_start: int,
    evidence_end: int,
    *,
    max_source_length: int,
) -> tuple[list[int], list[Piece], list[int], int, int] | None:
    query_ids = tokenizer.encode(query)
    context_budget = max_source_length - len(query_ids) - 1
    pieces = immutable_pieces(tokenizer, context)
    evidence_positions = span_piece_indices(pieces, evidence_start, evidence_end)
    if context_budget <= 0 or not evidence_positions or len(evidence_positions) > context_budget:
        return None

    if len(pieces) <= context_budget:
        left, right = 0, len(pieces)
    else:
        extra = context_budget - len(evidence_positions)
        left = max(0, evidence_positions[0] - extra // 2)
        right = min(len(pieces), left + context_budget)
        left = max(0, right - context_budget)

    while left <= evidence_positions[0] and right > evidence_positions[-1]:
        window_start = pieces[left].begin
        window_end = pieces[right - 1].end
        window = context[window_start:window_end]
        window_pieces = immutable_pieces(tokenizer, window)
        source_ids = tokenizer.encode_source(query, window)
        shifted_start = evidence_start - window_start
        shifted_end = evidence_end - window_start
        window_evidence = span_piece_indices(window_pieces, shifted_start, shifted_end)
        if len(source_ids) <= max_source_length and window_evidence:
            expected = [*query_ids, TOOLS_ID, *[piece.id for piece in window_pieces]]
            if source_ids != expected:
                raise ValueError("source format differs from NeedleTokenizer.encode_source")
            context_start = len(query_ids) + 1
            return source_ids, window_pieces, window_evidence, context_start, window_start

        left_room = evidence_positions[0] - left
        right_room = right - evidence_positions[-1] - 1
        if right_room >= left_room and right_room:
            right -= 1
        elif left_room:
            left += 1
        else:
            break
    return None


def prepare_squad2_item(
    item: dict[str, Any],
    tokenizer,
    *,
    example_index: int,
    max_source_length: int = 1024,
    max_target_length: int = 512,
) -> tuple[PreparedNeedleQA | None, str | None]:
    answers = item.get("answers", {})
    pairs = [(str(text), int(start)) for text, start in zip(answers.get("text", []), answers.get("answer_start", [])) if text]
    if not pairs:
        return None, "unanswerable"
    answer, answer_start = pairs[0]
    context = str(item.get("context", ""))
    answer_end = answer_start + len(answer)
    if answer_start < 0 or context[answer_start:answer_end] != answer:
        return None, "invalid_span"

    window = _evidence_window(
        tokenizer,
        str(item.get("question", "")),
        context,
        answer_start,
        answer_end,
        max_source_length=max_source_length,
    )
    if window is None:
        return None, "source_length_or_alignment"
    source_ids, context_pieces, evidence_positions, context_start, window_start = window
    shifted_start = answer_start - window_start
    shifted_end = answer_end - window_start
    window_context = context[window_start:]
    if not _covers_only_span_or_whitespace(window_context, context_pieces, evidence_positions, shifted_start, shifted_end):
        return None, "token_boundary_alignment"
    target_ids = [context_pieces[position].id for position in evidence_positions]
    if not target_ids or len(target_ids) + 1 > max_target_length:
        return None, "target_length"
    gold = [context_start + position for position in evidence_positions]
    return PreparedNeedleQA(
        source_ids=source_ids,
        target_ids=[*target_ids, EOS_ID],
        gold_copy_positions=[*gold, -1],
        context_start=context_start,
        example_index=example_index,
        turn_index=-1,
        evidence_start=answer_start,
        evidence_end=answer_end,
        window_start=window_start,
    ), None


def prepare_squad2_unanswerable(
    item: dict[str, Any],
    tokenizer,
    *,
    example_index: int,
    max_source_length: int = 1024,
) -> tuple[PreparedNeedleQA | None, str | None]:
    """Encode a real SQuAD2 negative without inventing an answer target."""
    if any(item.get("answers", {}).get("text", [])):
        return None, "answerable"
    query_ids = tokenizer.encode(str(item.get("question", "")))
    context_ids = tokenizer.encode(str(item.get("context", "")))
    context_budget = max_source_length - len(query_ids) - 1
    if context_budget <= 0 or not context_ids:
        return None, "source_length"
    context_ids = context_ids[:context_budget]
    context_start = len(query_ids) + 1
    return PreparedNeedleQA(
        source_ids=[*query_ids, TOOLS_ID, *context_ids],
        target_ids=[EOS_ID],
        gold_copy_positions=[-1],
        context_start=context_start,
        example_index=example_index,
        turn_index=-1,
        evidence_start=-1,
        evidence_end=-1,
        window_start=0,
    ), None


def coqa_query(history: list[tuple[str, str]], question: str, *, max_history_turns: int = 4) -> str:
    turns = history[-max_history_turns:] if max_history_turns else []
    if not turns:
        return question.strip()
    rendered = "\n".join(f"Q: {old_question}\nA: {old_answer}" for old_question, old_answer in turns)
    return f"Previous turns:\n{rendered}\nCurrent question: {question.strip()}"


def monotonic_rationale_alignment(target_ids: list[int], rationale_ids: list[int], rationale_positions: list[int]) -> list[int]:
    """Greedily align exact token IDs left-to-right; unmatched target tokens stay -1.

    This deliberately supervises only tokens copied verbatim and in order from
    the annotated rationale. Paraphrased, reordered, or absent tokens receive
    no source-position supervision.
    """
    aligned = [-1] * len(target_ids)
    cursor = 0
    for target_index, target_id in enumerate(target_ids):
        for rationale_index in range(cursor, len(rationale_ids)):
            if rationale_ids[rationale_index] == target_id:
                aligned[target_index] = rationale_positions[rationale_index]
                cursor = rationale_index + 1
                break
    return aligned


def prepare_coqa_turn(
    *,
    story: str,
    question: str,
    answer: str,
    rationale_start: int,
    rationale_end: int,
    history: list[tuple[str, str]],
    tokenizer,
    example_index: int,
    turn_index: int,
    max_history_turns: int = 4,
    max_source_length: int = 1024,
    max_target_length: int = 512,
) -> tuple[PreparedNeedleQA | None, str | None]:
    answer = answer.strip()
    if not answer or answer.lower() == "unknown":
        return None, "unanswerable"
    if not (0 <= rationale_start < rationale_end <= len(story)):
        return None, "invalid_rationale"
    target_ids = tokenizer.encode(answer)
    if not target_ids or len(target_ids) + 1 > max_target_length:
        return None, "target_length"

    window = None
    for history_turns in range(min(max_history_turns, len(history)), -1, -1):
        query = coqa_query(history, question, max_history_turns=history_turns)
        window = _evidence_window(
            tokenizer,
            query,
            story,
            rationale_start,
            rationale_end,
            max_source_length=max_source_length,
        )
        if window is not None:
            break
    if window is None:
        return None, "source_length_or_alignment"

    source_ids, context_pieces, rationale_piece_indices, context_start, window_start = window
    rationale_ids = [context_pieces[position].id for position in rationale_piece_indices]
    rationale_positions = [context_start + position for position in rationale_piece_indices]
    gold = monotonic_rationale_alignment(target_ids, rationale_ids, rationale_positions)
    return PreparedNeedleQA(
        source_ids=source_ids,
        target_ids=[*target_ids, EOS_ID],
        gold_copy_positions=[*gold, -1],
        context_start=context_start,
        example_index=example_index,
        turn_index=turn_index,
        evidence_start=rationale_start,
        evidence_end=rationale_end,
        window_start=window_start,
    ), None


def prepare_squad2_split(items: Iterable[dict[str, Any]], tokenizer, **kwargs) -> tuple[list[PreparedNeedleQA], dict[str, int]]:
    examples: list[PreparedNeedleQA] = []
    stats: dict[str, int] = {
        "input_rows": 0,
        "kept_rows": 0,
        "dropped_unanswerable": 0,
        "dropped_invalid_span": 0,
        "dropped_source_length_or_alignment": 0,
        "dropped_token_boundary_alignment": 0,
        "dropped_target_length": 0,
    }
    for index, item in enumerate(items):
        stats["input_rows"] += 1
        example, reason = prepare_squad2_item(item, tokenizer, example_index=index, **kwargs)
        if example is None:
            stats[f"dropped_{reason}"] = stats.get(f"dropped_{reason}", 0) + 1
            continue
        examples.append(example)
        _record_kept(stats, example)
    return examples, stats


def prepare_coqa_split(items: Iterable[dict[str, Any]], tokenizer, **kwargs) -> tuple[list[PreparedNeedleQA], dict[str, int]]:
    examples: list[PreparedNeedleQA] = []
    stats: dict[str, int] = {
        "input_conversations": 0,
        "input_turns": 0,
        "kept_rows": 0,
        "dropped_unanswerable": 0,
        "dropped_invalid_rationale": 0,
        "dropped_source_length_or_alignment": 0,
        "dropped_target_length": 0,
    }
    for conversation_index, item in enumerate(items):
        stats["input_conversations"] += 1
        story = str(item.get("story", ""))
        questions = item.get("questions", [])
        answers = item.get("answers", {})
        texts = answers.get("input_text", [])
        starts = answers.get("answer_start", [])
        ends = answers.get("answer_end", [])
        history: list[tuple[str, str]] = []
        for turn_index, question in enumerate(questions):
            stats["input_turns"] += 1
            answer = str(texts[turn_index]) if turn_index < len(texts) else ""
            start = int(starts[turn_index]) if turn_index < len(starts) else -1
            end = int(ends[turn_index]) if turn_index < len(ends) else -1
            example, reason = prepare_coqa_turn(
                story=story,
                question=str(question),
                answer=answer,
                rationale_start=start,
                rationale_end=end,
                history=history,
                tokenizer=tokenizer,
                example_index=conversation_index,
                turn_index=turn_index,
                **kwargs,
            )
            if example is None:
                stats[f"dropped_{reason}"] = stats.get(f"dropped_{reason}", 0) + 1
            else:
                examples.append(example)
                _record_kept(stats, example)
            history.append((str(question), answer))
    return examples, stats


def _record_kept(stats: dict[str, int], example: PreparedNeedleQA) -> None:
    stats["kept_rows"] += 1
    stats["source_tokens"] = stats.get("source_tokens", 0) + len(example.source_ids)
    stats["target_tokens"] = stats.get("target_tokens", 0) + len(example.target_ids)
    copyable = sum(position >= 0 for position in example.gold_copy_positions)
    stats["copyable_target_tokens"] = stats.get("copyable_target_tokens", 0) + copyable
    if copyable == 0:
        stats["zero_copy_rows"] = stats.get("zero_copy_rows", 0) + 1
