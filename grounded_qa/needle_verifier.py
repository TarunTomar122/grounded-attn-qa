"""Verification-only residual adapters for a frozen Needle reader."""

from __future__ import annotations

import torch
from torch import nn

from .needle_pointer import NeedlePointerModel


class ResidualAdapter(nn.Module):
    """A zero-initialized bottleneck update, active only in verifier mode."""

    def __init__(self, d_model: int, rank: int):
        super().__init__()
        self.down = nn.Linear(d_model, rank, bias=False)
        self.up = nn.Linear(rank, d_model, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up(torch.nn.functional.gelu(self.down(x)))


class NeedleVerifierAdapter(nn.Module):
    """Keeps the reader fixed while adding a small encoder-only verifier path."""

    def __init__(self, reader: NeedlePointerModel, rank: int = 32):
        super().__init__()
        self.reader = reader
        self.adapters = nn.ModuleList(ResidualAdapter(reader.cfg.d_model, rank) for _ in reader.encoder)
        for parameter in self.reader.parameters():
            parameter.requires_grad_(False)

    def encode(self, source_ids: torch.Tensor, source_valid: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(source_ids.shape[1], device=source_ids.device)
        x = self.reader.token_embedding(source_ids) * self.reader.cfg.embedding_scale
        for block, adapter in zip(self.reader.encoder, self.adapters):
            x = adapter(block(x, source_valid, positions))
        return self.reader.encoder_final_norm(x)

    @property
    def trainable_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
