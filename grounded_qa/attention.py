from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .rope import RotaryEmbedding


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1.0e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        return (x * torch.rsqrt(variance + self.eps)).to(x.dtype) * self.weight


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float, rope_theta: float):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout
        self.rope = RotaryEmbedding(self.head_dim, rope_theta)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor | None = None,
        *,
        key_valid: torch.Tensor | None = None,
        causal: bool = False,
        query_positions: torch.Tensor | None = None,
        key_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        key_value = query if key_value is None else key_value
        batch, query_len, d_model = query.shape
        key_len = key_value.shape[1]
        q = self.q_proj(query).view(batch, query_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key_value).view(batch, key_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(key_value).view(batch, key_len, self.n_heads, self.head_dim).transpose(1, 2)
        q, k = self.rope(q, k, query_positions, key_positions)

        attention_mask = None
        use_causal = causal and key_valid is None and query_len == key_len
        if key_valid is not None or (causal and not use_causal):
            valid = torch.ones((batch, query_len, key_len), dtype=torch.bool, device=query.device)
            if key_valid is not None:
                valid &= key_valid[:, None, :]
            if causal:
                valid &= torch.ones((query_len, key_len), dtype=torch.bool, device=query.device).tril()
            attention_mask = torch.zeros(
                (batch, 1, query_len, key_len),
                dtype=q.dtype,
                device=q.device,
            ).masked_fill(~valid[:, None], float("-inf"))

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=use_causal,
            scale=1.0 / math.sqrt(self.head_dim),
        )
        return self.out_proj(y.transpose(1, 2).contiguous().view(batch, query_len, d_model))
