from __future__ import annotations

import re
import unittest
from dataclasses import dataclass

import torch

from grounded_qa.needle_qa_data import (
    Piece,
    coqa_query,
    immutable_pieces,
    prepare_coqa_turn,
    prepare_squad2_item,
    span_piece_indices,
)
from scripts.prepare_needle_n2 import SOURCE_LENGTH, TARGET_LENGTH, tensorize


@dataclass
class FakeProtoPiece:
    id: int
    begin: int
    end: int


@dataclass
class FakeProto:
    pieces: list[FakeProtoPiece]


class FakeSentencePiece:
    def __init__(self, tokenizer: "FakeTokenizer"):
        self.tokenizer = tokenizer

    def encode(self, text: str, return_type: str | None = None) -> FakeProto:
        assert return_type == "proto"
        return FakeProto([
            FakeProtoPiece(self.tokenizer.token_id(match.group()), match.start(), match.end())
            for match in re.finditer(r"\w+|[^\w\s]", text)
        ])


class FakeTokenizer:
    def __init__(self):
        self.ids: dict[str, int] = {}
        self.sp = FakeSentencePiece(self)

    def token_id(self, token: str) -> int:
        key = token.lower()
        if key not in self.ids:
            self.ids[key] = len(self.ids) + 10
        return self.ids[key]

    def encode(self, text: str) -> list[int]:
        return [piece.id for piece in self.sp.encode(text, return_type="proto").pieces]

    def encode_source(self, query: str, context: str) -> list[int]:
        return [*self.encode(query), 5, *self.encode(context)]


class NeedleQADataTests(unittest.TestCase):
    def test_sentencepiece_utf8_byte_offsets_become_character_offsets(self) -> None:
        tokenizer = FakeTokenizer()
        tokenizer.sp.encode = lambda text, return_type=None: FakeProto([
            FakeProtoPiece(10, 0, 0),
            FakeProtoPiece(11, 0, len(text.encode("utf-8"))),
        ])
        self.assertEqual(immutable_pieces(tokenizer, "東"), [Piece(10, 0, 0), Piece(11, 0, 1)])

    def test_squad_uses_annotated_occurrence_and_deterministic_window(self) -> None:
        tokenizer = FakeTokenizer()
        context = "Paris came first. Paris came second."
        answer_start = context.rindex("Paris")
        item = {
            "question": "Where?",
            "context": context,
            "answers": {"text": ["Paris"], "answer_start": [answer_start]},
        }
        first, reason = prepare_squad2_item(item, tokenizer, example_index=7, max_source_length=8)
        second, _ = prepare_squad2_item(item, tokenizer, example_index=7, max_source_length=8)
        self.assertIsNone(reason)
        self.assertEqual(first, second)
        assert first is not None
        self.assertEqual(first.source_ids[: first.context_start], [*tokenizer.encode("Where?"), 5])
        self.assertEqual(first.source_ids[first.gold_copy_positions[0]], first.target_ids[0])
        self.assertEqual(first.evidence_start, answer_start)
        self.assertGreater(first.window_start, 0)
        self.assertEqual(first.target_ids[-1], 1)
        self.assertEqual(first.gold_copy_positions[-1], -1)

    def test_coqa_aligns_only_ordered_tokens_inside_rationale(self) -> None:
        tokenizer = FakeTokenizer()
        story = "Paris is the capital of France."
        example, reason = prepare_coqa_turn(
            story=story,
            question="What is the answer?",
            answer="Paris is in France",
            rationale_start=0,
            rationale_end=len(story),
            history=[(f"old question {index}", f"old answer {index}") for index in range(8)],
            tokenizer=tokenizer,
            example_index=2,
            turn_index=3,
            max_source_length=32,
        )
        self.assertIsNone(reason)
        assert example is not None
        self.assertGreaterEqual(example.gold_copy_positions[0], example.context_start)
        self.assertGreater(example.gold_copy_positions[1], example.gold_copy_positions[0])
        self.assertEqual(example.gold_copy_positions[2], -1)
        self.assertGreater(example.gold_copy_positions[3], example.gold_copy_positions[1])
        self.assertEqual(example.gold_copy_positions[-1], -1)

    def test_history_is_bounded_and_zero_means_no_history(self) -> None:
        history = [(f"q{index}", f"a{index}") for index in range(8)]
        query = coqa_query(history, "now?", max_history_turns=4)
        self.assertNotIn("q3", query)
        self.assertIn("q4", query)
        self.assertEqual(coqa_query(history, "now?", max_history_turns=0), "now?")

    def test_zero_width_sentencepiece_bytes_belong_to_span(self) -> None:
        pieces = [Piece(10, 5, 5), Piece(11, 5, 5), Piece(12, 5, 6)]
        self.assertEqual(span_piece_indices(pieces, 5, 6), [0, 1, 2])

    def test_squad_drops_answer_cut_from_inside_token(self) -> None:
        tokenizer = FakeTokenizer()
        example, reason = prepare_squad2_item(
            {
                "question": "What suffix?",
                "context": "prefixword",
                "answers": {"text": ["word"], "answer_start": [6]},
            },
            tokenizer,
            example_index=0,
        )
        self.assertIsNone(example)
        self.assertEqual(reason, "token_boundary_alignment")

    def test_fixed_tensor_contract(self) -> None:
        tokenizer = FakeTokenizer()
        context = "Alpha was born in Paris."
        start = context.index("Paris")
        example, _ = prepare_squad2_item(
            {
                "question": "Where was Alpha born?",
                "context": context,
                "answers": {"text": ["Paris"], "answer_start": [start]},
            },
            tokenizer,
            example_index=0,
        )
        assert example is not None
        tensors = tensorize([example])
        self.assertEqual(tensors["source_ids"].shape, (1, SOURCE_LENGTH))
        self.assertEqual(tensors["target_ids"].shape, (1, TARGET_LENGTH))
        for name in (
            "source_ids",
            "target_ids",
            "source_lengths",
            "target_lengths",
            "context_start",
            "gold_copy_positions",
        ):
            self.assertEqual(tensors[name].dtype, torch.int16)
        length = int(tensors["target_lengths"][0])
        self.assertEqual(int(tensors["target_ids"][0, length - 1]), 1)
        self.assertEqual(int(tensors["gold_copy_positions"][0, length - 1]), -1)


if __name__ == "__main__":
    unittest.main()
