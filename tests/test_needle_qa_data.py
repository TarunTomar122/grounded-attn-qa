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
    prepare_squad2_unanswerable,
    span_piece_indices,
)
from scripts.prepare_needle_n2 import SOURCE_LENGTH, TARGET_LENGTH, tensorize
from scripts.prepare_needle_n3 import cross_pair_negatives
from scripts.prepare_needle_n3_entity import prepare_entity_binding_pairs, prepare_relation_binding_pairs
from scripts.prepare_needle_n3_official import partition_tensor_rows, select_tensor_rows


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

    def test_squad_unanswerable_has_no_copy_target(self) -> None:
        tokenizer = FakeTokenizer()
        example, reason = prepare_squad2_unanswerable(
            {"question": "Who signed it?", "context": "The record names no signer.", "answers": {"text": [], "answer_start": []}},
            tokenizer,
            example_index=3,
        )
        self.assertIsNone(reason)
        assert example is not None
        self.assertEqual(example.target_ids, [1])
        self.assertEqual(example.gold_copy_positions, [-1])
        self.assertEqual(example.source_ids[example.context_start - 1], 5)

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

    def test_cross_pair_negative_keeps_question_and_replaces_context(self) -> None:
        positives = {
            "source_ids": torch.tensor([
                [10, 5, 20, 21, 0, 0],
                [11, 5, 30, 31, 0, 0],
                [12, 5, 40, 41, 0, 0],
                [13, 5, 50, 51, 0, 0],
            ], dtype=torch.int16),
            "target_ids": torch.tensor([[20, 1], [30, 1], [40, 1], [50, 1]], dtype=torch.int16),
            "source_lengths": torch.tensor([4, 4, 4, 4], dtype=torch.int16),
            "target_lengths": torch.tensor([2, 2, 2, 2], dtype=torch.int16),
            "context_start": torch.tensor([2, 2, 2, 2], dtype=torch.int16),
            "gold_copy_positions": torch.tensor([[2, -1], [2, -1], [2, -1], [2, -1]], dtype=torch.int16),
            "example_index": torch.tensor([0, 1, 2, 3], dtype=torch.int32),
            "turn_index": torch.zeros(4, dtype=torch.int16),
            "evidence_start": torch.zeros(4, dtype=torch.int32),
            "evidence_end": torch.ones(4, dtype=torch.int32),
            "window_start": torch.zeros(4, dtype=torch.int32),
        }

        negatives = cross_pair_negatives(positives, count=4)

        for row, source_index in enumerate(negatives["example_index"].tolist()):
            self.assertTrue(torch.equal(negatives["source_ids"][row, :2], positives["source_ids"][source_index, :2]))
            self.assertFalse(torch.equal(negatives["source_ids"][row, 2:4], positives["source_ids"][source_index, 2:4]))
        self.assertTrue(torch.equal(negatives["target_ids"][:, 0], torch.ones(4, dtype=torch.int16)))
        self.assertTrue(torch.equal(negatives["target_lengths"], torch.ones(4, dtype=torch.int16)))
        self.assertTrue((negatives["gold_copy_positions"] == -1).all())

    def test_entity_binding_pairs_keep_context_and_mark_missing_subject_unanswerable(self) -> None:
        positive, negative = prepare_entity_binding_pairs(3, FakeTokenizer(), seed=7)

        self.assertEqual(len(positive), 3)
        self.assertEqual(len(negative), 3)
        for answerable, unanswerable in zip(positive, negative):
            self.assertGreater(len(answerable.target_ids), 1)
            self.assertEqual(unanswerable.target_ids, [1])
            self.assertTrue(all(position == -1 for position in unanswerable.gold_copy_positions))
            self.assertEqual(
                answerable.source_ids[answerable.context_start :],
                unanswerable.source_ids[unanswerable.context_start :],
            )

    def test_relation_binding_pairs_keep_subject_and_remove_requested_fact(self) -> None:
        positive, negative = prepare_relation_binding_pairs(3, FakeTokenizer(), seed=11)

        for answerable, unanswerable in zip(positive, negative):
            self.assertGreater(len(answerable.target_ids), 1)
            self.assertEqual(unanswerable.target_ids, [1])
            self.assertTrue(all(position == -1 for position in unanswerable.gold_copy_positions))
            self.assertEqual(
                answerable.source_ids[: answerable.context_start],
                unanswerable.source_ids[: unanswerable.context_start],
            )
            self.assertNotEqual(
                answerable.source_ids[answerable.context_start :],
                unanswerable.source_ids[unanswerable.context_start :],
            )

    def test_tensor_row_selection_is_deterministic_and_non_mutating(self) -> None:
        rows = {"source_ids": torch.arange(12).reshape(4, 3), "answerable": torch.ones(4, dtype=torch.bool)}
        first = select_tensor_rows(rows, 3, seed=5)
        second = select_tensor_rows(rows, 3, seed=5)
        self.assertTrue(torch.equal(first["source_ids"], second["source_ids"]))
        self.assertTrue(torch.equal(first["answerable"], second["answerable"]))
        self.assertEqual(first["source_ids"].shape, (3, 3))
        self.assertTrue(rows["answerable"].all())

    def test_train_development_partition_is_deterministic_and_disjoint(self) -> None:
        rows = {"source_ids": torch.arange(30).reshape(10, 3), "answerable": torch.ones(10, dtype=torch.bool)}
        train, development = partition_tensor_rows(rows, 0.2, seed=13)
        repeated_train, repeated_development = partition_tensor_rows(rows, 0.2, seed=13)

        self.assertTrue(torch.equal(train["source_ids"], repeated_train["source_ids"]))
        self.assertTrue(torch.equal(development["source_ids"], repeated_development["source_ids"]))
        self.assertEqual(len(train["source_ids"]), 8)
        self.assertEqual(len(development["source_ids"]), 2)
        self.assertFalse(set(train["source_ids"][:, 0].tolist()) & set(development["source_ids"][:, 0].tolist()))


if __name__ == "__main__":
    unittest.main()
