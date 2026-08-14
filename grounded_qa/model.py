from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .attention import MultiHeadAttention, RMSNorm
from .config import ModelConfig
from .pointer import PointerGenerator


class EncoderBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm = RMSNorm(cfg.d_model, cfg.rmsnorm_eps)
        self.attention = MultiHeadAttention(cfg.d_model, cfg.n_heads, cfg.dropout, cfg.rope_theta)

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        return x + self.attention(self.norm(x), key_valid=valid)


class DecoderBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.self_norm = RMSNorm(cfg.d_model, cfg.rmsnorm_eps)
        self.self_attention = MultiHeadAttention(cfg.d_model, cfg.n_heads, cfg.dropout, cfg.rope_theta)
        self.cross_norm = RMSNorm(cfg.d_model, cfg.rmsnorm_eps)
        self.cross_attention = MultiHeadAttention(cfg.d_model, cfg.n_heads, cfg.dropout, cfg.rope_theta)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        target_valid: torch.Tensor,
        source_valid: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.self_attention(self.self_norm(x), key_valid=target_valid, causal=True)
        return x + self.cross_attention(self.cross_norm(x), memory, key_valid=source_valid)


@dataclass
class GroundedOutput:
    vocab_logits: torch.Tensor
    copy_position_probs: torch.Tensor
    p_gen: torch.Tensor
    pointer_context: torch.Tensor
    answerability_logits: torch.Tensor
    decoder_hidden: torch.Tensor
    memory: torch.Tensor
    stop_probability: torch.Tensor
    answer_start_logits: torch.Tensor

    def final_distribution(
        self,
        source_ids: torch.Tensor,
        *,
        copy_only: bool = False,
        eos_id: int | None = None,
    ) -> torch.Tensor:
        copy_distribution = PointerGenerator.copy_distribution(
            self.copy_position_probs,
            source_ids,
            self.vocab_logits.shape[-1],
        )
        if copy_only:
            if eos_id is None:
                raise ValueError("copy-only decoding needs eos_id")
            distribution = copy_distribution * (1 - self.stop_probability[..., None])
            distribution[..., eos_id] += self.stop_probability
            return distribution
        vocab_distribution = self.vocab_logits.softmax(dim=-1)
        return self.p_gen[..., None] * vocab_distribution + (1 - self.p_gen[..., None]) * copy_distribution


class GroundedPointerGenerator(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.token_type_embedding = nn.Embedding(cfg.num_token_types, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)
        self.encoder = nn.ModuleList(EncoderBlock(cfg) for _ in range(cfg.encoder_layers))
        self.decoder = nn.ModuleList(DecoderBlock(cfg) for _ in range(cfg.decoder_layers))
        self.encoder_final_norm = RMSNorm(cfg.d_model, cfg.rmsnorm_eps)
        self.decoder_final_norm = RMSNorm(cfg.d_model, cfg.rmsnorm_eps)
        self.pointer = PointerGenerator(cfg.d_model)
        self.stop_head = nn.Linear(cfg.d_model * 2, 1)
        self.answerability = nn.Linear(cfg.d_model, 1)
        self.answer_start_head = nn.Linear(cfg.d_model, 1)
        self.answer_start_global_head = nn.Linear(cfg.d_model * 2, 1)
        self.apply(self._init)

    def _init(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

    def encode(
        self,
        source_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        source_valid: torch.Tensor,
    ) -> torch.Tensor:
        x = self.dropout(self.token_embedding(source_ids) + self.token_type_embedding(token_type_ids))
        for block in self.encoder:
            x = block(x, source_valid)
        return self.encoder_final_norm(x)

    def decode(
        self,
        decoder_input_ids: torch.Tensor,
        memory: torch.Tensor,
        source_ids: torch.Tensor,
        context_mask: torch.Tensor,
        source_valid: torch.Tensor,
        target_valid: torch.Tensor,
    ) -> GroundedOutput:
        x = self.dropout(self.token_embedding(decoder_input_ids))
        for block in self.decoder:
            x = block(x, memory, target_valid, source_valid)
        x = self.decoder_final_norm(x)
        vocab_logits, copy_probs, p_gen, pointer_context = self.pointer(
            x,
            memory,
            source_ids,
            context_mask & source_valid,
            self.token_embedding(decoder_input_ids),
            self.token_embedding.weight,
        )
        stop_probability = torch.sigmoid(self.stop_head(torch.cat((x, pointer_context), dim=-1))).squeeze(-1)
        if self.cfg.answer_start_mode == "global":
            global_memory = memory[:, :1].expand(-1, memory.shape[1], -1)
            answer_start_logits = self.answer_start_global_head(torch.cat((memory, global_memory), dim=-1)).squeeze(-1)
        else:
            answer_start_logits = self.answer_start_head(memory).squeeze(-1)
        return GroundedOutput(
            vocab_logits=vocab_logits,
            copy_position_probs=copy_probs,
            p_gen=p_gen,
            pointer_context=pointer_context,
            answerability_logits=self.answerability(memory[:, 0]).squeeze(-1),
            decoder_hidden=x,
            memory=memory,
            stop_probability=stop_probability,
            answer_start_logits=answer_start_logits,
        )

    def forward(
        self,
        source_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        source_valid: torch.Tensor,
        context_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        target_valid: torch.Tensor,
    ) -> GroundedOutput:
        memory = self.encode(source_ids, token_type_ids, source_valid)
        return self.decode(
            decoder_input_ids,
            memory,
            source_ids,
            context_mask,
            source_valid,
            target_valid,
        )

    def parameter_breakdown(self) -> dict[str, int]:
        groups = {
            "shared_token_embedding": self.token_embedding,
            "token_type_embedding": self.token_type_embedding,
            "encoder": self.encoder,
            "decoder": self.decoder,
            "normalization": nn.ModuleList([self.encoder_final_norm, self.decoder_final_norm]),
            "pointer": self.pointer,
            "stop_head": self.stop_head,
            "answerability": self.answerability,
            "answer_start_head": self.answer_start_head,
            "answer_start_global_head": self.answer_start_global_head,
        }
        return {name: sum(p.numel() for p in module.parameters()) for name, module in groups.items()}

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
