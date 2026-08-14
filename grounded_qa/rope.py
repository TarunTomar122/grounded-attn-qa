from __future__ import annotations

import torch
from torch import nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10_000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        positions_q: torch.Tensor | None = None,
        positions_k: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        positions_q = self._positions(q.shape[-2], q.device, positions_q)
        positions_k = self._positions(k.shape[-2], k.device, positions_k)
        q = self._rotate(q, positions_q)
        k = self._rotate(k, positions_k)
        return q, k

    def _positions(self, length: int, device: torch.device, positions: torch.Tensor | None) -> torch.Tensor:
        if positions is None:
            return torch.arange(length, device=device)
        return positions.to(device=device)

    def _rotate(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        freqs = torch.einsum("...t,d->...td", positions.float(), self.inv_freq.to(positions.device))
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        if positions.ndim == 1:
            cos = cos[None, None, :, :]
            sin = sin[None, None, :, :]
        else:
            cos = cos[:, None, :, :]
            sin = sin[:, None, :, :]
        return x * cos + rotate_half(x) * sin
