import unittest

import torch

from scripts.train import _gold_answer_span, _gold_answer_tokens, _length_bucket


class TrainMetricTests(unittest.TestCase):
    def test_gold_tokens_remove_each_row_eos_before_padding(self) -> None:
        batch = {
            "target_ids": torch.tensor([
                [10, 11, 2, 0],
                [12, 2, 0, 0],
            ]),
            "target_valid": torch.tensor([
                [True, True, True, False],
                [True, True, False, False],
            ]),
        }
        self.assertEqual(_gold_answer_tokens(batch, 0, 2), [10, 11])
        self.assertEqual(_gold_answer_tokens(batch, 1, 2), [12])

    def test_answer_length_buckets(self) -> None:
        self.assertEqual(_length_bucket(2), "1-2")
        self.assertEqual(_length_bucket(4), "3-4")
        self.assertEqual(_length_bucket(6), "5-6")
        self.assertEqual(_length_bucket(7), "7-10")
        self.assertEqual(_length_bucket(11), "11+")

    def test_gold_span_uses_aligned_source_positions(self) -> None:
        batch = {
            "gold_copy_positions": torch.tensor([[8, 9, -1]]),
        }
        self.assertEqual(_gold_answer_span(batch, 0), [8, 9])


if __name__ == "__main__":
    unittest.main()
