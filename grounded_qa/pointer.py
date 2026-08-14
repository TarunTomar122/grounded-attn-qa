from __future__ import annotations

import math

import torch
from torch import nn


class PointerGenerator(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.pointer_q = nn.Linear(d_model, d_model, bias=False)
        self.pointer_k = nn.Linear(d_model, d_model, bias=False)
        self.gate = nn.Linear(d_model * 3, 1)

    def forward(
        self,
        decoder_state: torch.Tensor,
        memory: torch.Tensor,
        source_ids: torch.Tensor,
        context_mask: torch.Tensor,
        previous_embedding: torch.Tensor,
        vocab_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        scores = torch.einsum(
            "btd,bsd->bts",
            self.pointer_q(decoder_state),
            self.pointer_k(memory),
        ) / math.sqrt(memory.shape[-1])
        if not context_mask.any(dim=-1).all():
            raise ValueError("Every example needs at least one copyable context token")
        scores = scores.masked_fill(~context_mask[:, None, :], float("-inf"))
        copy_position_probs = scores.softmax(dim=-1)
        pointer_context = torch.einsum("bts,bsd->btd", copy_position_probs, memory)
        p_gen = torch.sigmoid(
            self.gate(torch.cat((decoder_state, pointer_context, previous_embedding), dim=-1))
        ).squeeze(-1)
        vocab_logits = decoder_state @ vocab_weight.transpose(0, 1)
        return vocab_logits, copy_position_probs, p_gen, pointer_context

    @staticmethod
    def copy_distribution(
        copy_position_probs: torch.Tensor,
        source_ids: torch.Tensor,
        vocab_size: int,
    ) -> torch.Tensor:
        distribution = copy_position_probs.new_zeros(
            (*copy_position_probs.shape[:2], vocab_size)
        )
        ids = source_ids[:, None, :].expand_as(copy_position_probs)
        return distribution.scatter_add(-1, ids, copy_position_probs)
