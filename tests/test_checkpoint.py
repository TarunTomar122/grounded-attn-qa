from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from grounded_qa.checkpoint import load_checkpoint, save_checkpoint
from grounded_qa.config import ModelConfig
from grounded_qa.model import GroundedPointerGenerator


class CheckpointTests(unittest.TestCase):
    def test_checkpoint_restores_model_and_optimizer_step(self) -> None:
        cfg = ModelConfig(vocab_size=32, d_model=32, n_heads=4, encoder_layers=1, decoder_layers=1, dropout=0.0)
        model = GroundedPointerGenerator(cfg)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        optimizer.zero_grad()
        model.token_embedding.weight.sum().backward()
        optimizer.step()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            save_checkpoint(path, model, optimizer, None, step=7, tokens_seen=123, phase="A", config=cfg, tokenizer_path="tokenizer.json", seed=42)
            restored = GroundedPointerGenerator(cfg)
            restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
            state = load_checkpoint(path, restored, restored_optimizer)
            self.assertEqual(state["step"], 7)
            self.assertEqual(state["tokens_seen"], 123)
            torch.testing.assert_close(restored.token_embedding.weight, model.token_embedding.weight)

    def test_checkpoint_rng_state_loads_when_checkpoint_is_mapped_to_accelerator(self) -> None:
        cfg = ModelConfig(vocab_size=32, d_model=32, n_heads=4, encoder_layers=1, decoder_layers=1, dropout=0.0)
        model = GroundedPointerGenerator(cfg)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            save_checkpoint(path, model, None, None, step=1, tokens_seen=0, phase="A", config=cfg, tokenizer_path="tokenizer.json", seed=42)
            restored = GroundedPointerGenerator(cfg)
            load_checkpoint(path, restored, device="mps" if torch.backends.mps.is_available() else "cpu")


if __name__ == "__main__":
    unittest.main()
