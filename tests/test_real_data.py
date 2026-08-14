from __future__ import annotations

import unittest

from grounded_qa.data import encode_row
from grounded_qa.real_data import squad2_rows_from_items
from grounded_qa.tokenizer import load_tokenizer


class RealDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tokenizer, cls.info = load_tokenizer("artifacts/phase-a1-v6/tokenizer.json")

    def test_answer_start_maps_to_exact_source_tokens(self) -> None:
        context = "Alpha was born in Smiljan in 1856."
        answer = "Smiljan"
        start = context.index(answer)
        rows, stats = squad2_rows_from_items(
            [{
                "id": "one",
                "title": "test",
                "question": "Where was Alpha born?",
                "context": context,
                "answers": {"text": [answer], "answer_start": [start]},
            }],
            split="train",
            tokenizer=self.tokenizer,
        )
        self.assertEqual(stats["kept"], 1)
        encoded = encode_row(rows[0], self.tokenizer, self.info, max_source_length=512, max_target_length=64)
        assert encoded is not None
        positions = encoded.gold_copy_positions[:-1]
        self.assertEqual([encoded.source_ids[position] for position in positions], encoded.target_ids[:-1])
        self.assertEqual(rows[0]["answer_start"], start)
        self.assertEqual(rows[0]["answer_end"], start + len(answer))

    def test_long_context_is_windowed_and_validation_is_deterministic(self) -> None:
        context = " ".join(f"word{index}" for index in range(120))
        answer = "word87"
        start = context.index(answer)
        item = {
            "id": "long",
            "title": "test",
            "question": "Which word is the answer?",
            "context": context,
            "answers": {"text": [answer], "answer_start": [start]},
        }
        first, _ = squad2_rows_from_items([item], split="validation", tokenizer=self.tokenizer, source_length=64, validation=True)
        second, _ = squad2_rows_from_items([item], split="validation", tokenizer=self.tokenizer, source_length=64, validation=True)
        self.assertEqual(first, second)
        self.assertTrue(first[0]["metadata"]["windowed"])
        encoded = encode_row(first[0], self.tokenizer, self.info, max_source_length=64, max_target_length=64)
        self.assertIsNotNone(encoded)

    def test_unanswerable_and_bad_offsets_are_counted(self) -> None:
        rows, stats = squad2_rows_from_items(
            [
                {"id": "no", "question": "?", "context": "text", "answers": {"text": [], "answer_start": []}},
                {"id": "bad", "question": "?", "context": "text", "answers": {"text": ["text"], "answer_start": [1]}},
            ],
            split="validation",
            tokenizer=self.tokenizer,
        )
        self.assertEqual(rows, [])
        self.assertEqual(stats["unanswerable"], 1)
        self.assertEqual(stats["invalid_span"], 1)


if __name__ == "__main__":
    unittest.main()
