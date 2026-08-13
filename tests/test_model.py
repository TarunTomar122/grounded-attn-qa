from __future__ import annotations

import unittest

import torch

from grounded_qa.config import ModelConfig
from grounded_qa.losses import grounded_loss
from grounded_qa.model import GroundedPointerGenerator
from grounded_qa.pointer import PointerGenerator
from grounded_qa.rope import RotaryEmbedding


class ModelTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.cfg = ModelConfig(
            vocab_size=64,
            d_model=32,
            n_heads=4,
            encoder_layers=2,
            decoder_layers=2,
            dropout=0.0,
            source_length=16,
            target_length=8,
        )
        self.model = GroundedPointerGenerator(self.cfg).to("cpu").eval()
        self.source = torch.tensor([[3, 5, 11, 12, 4, 6, 11, 13, 14, 2]])
        self.types = torch.tensor([[0, 0, 0, 0, 0, 1, 1, 1, 1, 0]])
        self.valid = torch.ones_like(self.source, dtype=torch.bool)
        self.context = torch.tensor([[False, False, False, False, False, False, True, True, True, False]])

    def test_encoder_can_read_right_and_decoder_cannot_read_future(self) -> None:
        memory = self.model.encode(self.source, self.types, self.valid)
        changed = self.source.clone()
        changed[0, 8] = 31
        changed_memory = self.model.encode(changed, self.types, self.valid)
        self.assertFalse(torch.allclose(memory[:, 0], changed_memory[:, 0]))

        first = torch.tensor([[1, 10, 20]])
        second = torch.tensor([[1, 10, 21]])
        first_out = self.model.decode(first, memory, self.source, self.context, self.valid, torch.ones_like(first, dtype=torch.bool))
        second_out = self.model.decode(second, memory, self.source, self.context, self.valid, torch.ones_like(second, dtype=torch.bool))
        torch.testing.assert_close(first_out.decoder_hidden[:, 0], second_out.decoder_hidden[:, 0])

    def test_pointer_only_sees_context_and_final_distribution_normalizes(self) -> None:
        output = self.model(
            self.source,
            self.types,
            self.valid,
            self.context,
            torch.tensor([[1, 11, 14]]),
            torch.ones((1, 3), dtype=torch.bool),
        )
        self.assertTrue(torch.equal(output.copy_position_probs[..., :6], torch.zeros_like(output.copy_position_probs[..., :6])))
        torch.testing.assert_close(output.final_distribution(self.source).sum(-1), torch.ones(1, 3), atol=1e-5, rtol=1e-5)

    def test_copy_only_distribution_reserves_probability_for_eos(self) -> None:
        output = self.model(
            self.source,
            self.types,
            self.valid,
            self.context,
            torch.tensor([[1, 11, 14]]),
            torch.ones((1, 3), dtype=torch.bool),
        )
        distribution = output.final_distribution(self.source, copy_only=True, eos_id=self.cfg.eos_id)
        torch.testing.assert_close(distribution.sum(-1), torch.ones(1, 3), atol=1e-5, rtol=1e-5)
        self.assertTrue(torch.all(distribution[..., self.cfg.eos_id] > 0))

    def test_stop_head_consumes_pointer_context(self) -> None:
        output = self.model(
            self.source,
            self.types,
            self.valid,
            self.context,
            torch.tensor([[1, 11, 14]]),
            torch.ones((1, 3), dtype=torch.bool),
        )
        self.assertEqual(self.model.stop_head.in_features, self.cfg.d_model * 2)
        self.assertEqual(output.stop_probability.shape, output.copy_position_probs.shape[:2])

    def test_repeated_source_tokens_are_summed(self) -> None:
        probs = torch.tensor([[[0.2, 0.3, 0.5]]])
        ids = torch.tensor([[4, 4, 8]])
        distribution = PointerGenerator.copy_distribution(probs, ids, 16)
        torch.testing.assert_close(distribution[0, 0, 4], torch.tensor(0.5))
        torch.testing.assert_close(distribution[0, 0, 8], torch.tensor(0.5))

    def test_tied_embedding_and_output_projection_share_storage(self) -> None:
        output = self.model(
            self.source,
            self.types,
            self.valid,
            self.context,
            torch.tensor([[1, 11]]),
            torch.ones((1, 2), dtype=torch.bool),
        )
        expected = output.decoder_hidden @ self.model.token_embedding.weight.transpose(0, 1)
        torch.testing.assert_close(output.vocab_logits, expected)

    def test_unanswerable_has_no_sequence_loss_but_has_answerability_gradient(self) -> None:
        output = self.model(
            self.source,
            self.types,
            self.valid,
            self.context,
            torch.tensor([[1, 2, 2]]),
            torch.ones((1, 3), dtype=torch.bool),
        )
        loss = grounded_loss(
            output,
            self.source,
            torch.tensor([[2, 2, 2]]),
            torch.zeros((1, 3), dtype=torch.bool),
            torch.tensor([False]),
        )
        self.assertEqual(loss.sequence.item(), 0.0)
        loss.total.backward()
        self.assertIsNotNone(self.model.answerability.weight.grad)
        self.assertGreater(self.model.answerability.weight.grad.abs().sum().item(), 0.0)

    def test_pointer_position_loss_uses_gold_source_occurrence(self) -> None:
        output = self.model(
            self.source,
            self.types,
            self.valid,
            self.context,
            torch.tensor([[1, 11, 13]]),
            torch.ones((1, 3), dtype=torch.bool),
        )
        loss = grounded_loss(
            output,
            self.source,
            torch.tensor([[11, 13, 2]]),
            torch.ones((1, 3), dtype=torch.bool),
            torch.tensor([True]),
            lambda_answerability=0.0,
            copy_only=True,
            eos_id=self.cfg.eos_id,
            gold_copy_positions=torch.tensor([[6, 7, -1]]),
            lambda_pointer_position=1.0,
        )
        self.assertGreater(loss.pointer_position.item(), 0.0)
        self.assertGreater(loss.total.item(), loss.sequence.item())

    def test_rope_accepts_configured_head_dimension(self) -> None:
        rope = RotaryEmbedding(self.cfg.head_dim)
        q = torch.randn(2, self.cfg.n_heads, 5, self.cfg.head_dim)
        k = torch.randn_like(q)
        rotated_q, rotated_k = rope(q, k)
        self.assertEqual(rotated_q.shape, q.shape)
        self.assertEqual(rotated_k.shape, k.shape)

    def test_parameter_budget_is_close_to_plan(self) -> None:
        full = GroundedPointerGenerator(ModelConfig())
        self.assertGreater(full.n_params(), 23_000_000)
        self.assertLess(full.n_params(), 24_000_000)


if __name__ == "__main__":
    unittest.main()
