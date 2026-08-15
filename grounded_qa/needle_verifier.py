"""Verification-only residual adapters for a frozen Needle reader."""

from __future__ import annotations

import torch
from torch import nn

from .needle_pointer import NeedlePointerModel, candidate_span_features


def joint_verifier_logits(
    reader: NeedlePointerModel,
    verifier: nn.Module,
    source_ids: torch.Tensor,
    source_valid: torch.Tensor,
    question_mask: torch.Tensor,
    candidate_mask: torch.Tensor,
) -> torch.Tensor:
    """Classify a candidate while allowing verifier gradients into the shared reader."""
    memory = reader.encode(source_ids, source_valid)
    features = candidate_span_features(memory, source_valid, question_mask, candidate_mask)
    return verifier(features)


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

    def __init__(self, reader: NeedlePointerModel, rank: int = 32, *, decoder: bool = False):
        super().__init__()
        self.reader = reader
        self.adapters = nn.ModuleList(ResidualAdapter(reader.cfg.d_model, rank) for _ in reader.encoder)
        self.decoder_adapters = nn.ModuleList(ResidualAdapter(reader.cfg.d_model, rank) for _ in reader.decoder) if decoder else None
        for parameter in self.reader.parameters():
            parameter.requires_grad_(False)

    def encode(self, source_ids: torch.Tensor, source_valid: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(source_ids.shape[1], device=source_ids.device)
        x = self.reader.token_embedding(source_ids) * self.reader.cfg.embedding_scale
        for block, adapter in zip(self.reader.encoder, self.adapters):
            x = adapter(block(x, source_valid, positions))
        return self.reader.encoder_final_norm(x)

    def verify(self, source_ids: torch.Tensor, source_valid: torch.Tensor, evidence_valid: torch.Tensor) -> torch.Tensor:
        """One decoder token asks the frozen reader's cross-attention to assess evidence."""
        if self.decoder_adapters is None:
            raise ValueError("verify requires decoder=True")
        memory = self.encode(source_ids, source_valid)
        target = torch.full((len(source_ids), 1), self.reader.cfg.bos_id, dtype=torch.long, device=source_ids.device)
        target_valid = torch.ones_like(target, dtype=torch.bool)
        target_positions = torch.zeros(1, dtype=torch.long, device=source_ids.device)
        source_positions = torch.arange(source_ids.shape[1], device=source_ids.device)
        x = self.reader.token_embedding(target) * self.reader.cfg.embedding_scale
        for block, adapter in zip(self.reader.decoder, self.decoder_adapters):
            x = adapter(block(x, memory, target_valid, evidence_valid, target_positions, source_positions))
        return self.reader.decoder_final_norm(x).squeeze(1)

    @property
    def trainable_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
