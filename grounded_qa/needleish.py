from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class NeedleConfig:
    model_name: str = "needleish26m_v1"
    vocab_size: int = 8196
    d_model: int = 512
    encoder_layers: int = 12
    decoder_layers: int = 8
    query_heads: int = 8
    kv_heads: int = 4
    head_dim: int = 64
    rope_theta: float = 10_000.0
    rmsnorm_eps: float = 1.0e-6
    source_length: int = 512
    target_length: int = 256
    pad_id: int = 0
    eos_id: int = 1
    bos_id: int = 2
    unk_id: int = 3
    query_id: int = 8192
    context_id: int = 8193
    reasoning_id: int = 8194
    answer_id: int = 8195
    dropout: float = 0.0
    embedding_scale: float = 1.0
    cross_attention_rope: bool = True

    def __post_init__(self) -> None:
        if self.d_model != self.query_heads * self.head_dim:
            raise ValueError("d_model must equal query_heads * head_dim")
        if self.query_heads % self.kv_heads:
            raise ValueError("query_heads must be divisible by kv_heads")
        if self.head_dim % 2:
            raise ValueError("RoPE requires an even head dimension")

    @classmethod
    def public_checkpoint(cls) -> "NeedleConfig":
        """Exact trainable architecture released as Cactus-Compute/needle."""
        return cls(
            model_name="cactus_needle_26m",
            vocab_size=8192,
            source_length=1024,
            target_length=512,
            embedding_scale=math.sqrt(512),
            cross_attention_rope=False,
        )

    def to_dict(self) -> dict[str, int | float | str | bool]:
        return asdict(self)


class ZCRMSNorm(nn.Module):
    """Needle's zero-centred RMSNorm: (1 + scale) * x / RMS(x)."""

    def __init__(self, dim: int, eps: float = 1.0e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x_float = x.float()
        rms = torch.sqrt(x_float.square().mean(dim=-1, keepdim=True) + self.eps)
        return ((1.0 + self.scale) * x_float / rms).to(dtype)


class RoPE(nn.Module):
    def __init__(self, head_dim: int, max_length: int, theta: float):
        super().__init__()
        frequencies = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        angles = torch.outer(torch.arange(max_length).float(), frequencies)
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        cos = self.cos[positions].to(x.dtype)[None, None]
        sin = self.sin[positions].to(x.dtype)[None, None]
        half = x.shape[-1] // 2
        first, second = x[..., :half], x[..., half:]
        return torch.cat((first * cos - second * sin, second * cos + first * sin), dim=-1)


class GroupedQueryAttention(nn.Module):
    def __init__(self, cfg: NeedleConfig, *, causal: bool = False):
        super().__init__()
        self.query_heads = cfg.query_heads
        self.kv_heads = cfg.kv_heads
        self.head_dim = cfg.head_dim
        self.causal = causal
        kv_dim = cfg.kv_heads * cfg.head_dim
        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, kv_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, kv_dim, bias=False)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.q_norm = ZCRMSNorm(cfg.head_dim, cfg.rmsnorm_eps)
        self.k_norm = ZCRMSNorm(cfg.head_dim, cfg.rmsnorm_eps)
        self.rope = RoPE(cfg.head_dim, max(cfg.source_length, cfg.target_length), cfg.rope_theta)
        self.last_logit_std = 0.0
        self.collect_stats = False

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        *,
        key_valid: torch.Tensor,
        query_positions: torch.Tensor,
        key_positions: torch.Tensor,
        causal: bool | None = None,
        use_rope: bool = True,
    ) -> torch.Tensor:
        batch, query_length, _ = query.shape
        key_length = key_value.shape[1]
        q = self.q_proj(query).view(batch, query_length, self.query_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key_value).view(batch, key_length, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(key_value).view(batch, key_length, self.kv_heads, self.head_dim).transpose(1, 2)
        q = self.q_norm(q)
        k = self.k_norm(k)
        if use_rope:
            q = self.rope(q, query_positions)
            k = self.rope(k, key_positions)
        if self.query_heads != self.kv_heads:
            repeats = self.query_heads // self.kv_heads
            k = k.repeat_interleave(repeats, dim=1)
            v = v.repeat_interleave(repeats, dim=1)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if self.collect_stats:
            self.last_logit_std = float(scores.detach().float().std().cpu())
        valid = key_valid[:, None, None, :].expand(batch, 1, query_length, key_length)
        use_causal = self.causal if causal is None else causal
        if use_causal:
            causal_mask = torch.ones((query_length, key_length), dtype=torch.bool, device=query.device).tril()
            valid = valid & causal_mask[None, None]
        scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
        attention = F.softmax(scores.float(), dim=-1).to(scores.dtype)
        output = torch.matmul(attention, v).transpose(1, 2).reshape(batch, query_length, -1)
        return self.out_proj(output)


class EncoderBlock(nn.Module):
    def __init__(self, cfg: NeedleConfig):
        super().__init__()
        self.input_layernorm = ZCRMSNorm(cfg.d_model, cfg.rmsnorm_eps)
        self.self_attn = GroupedQueryAttention(cfg)
        self.attn_gate = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, valid: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        normalized = self.input_layernorm(x)
        update = self.self_attn(normalized, normalized, key_valid=valid, query_positions=positions, key_positions=positions)
        return x + torch.sigmoid(self.attn_gate) * update


class DecoderBlock(nn.Module):
    def __init__(self, cfg: NeedleConfig):
        super().__init__()
        self.input_layernorm = ZCRMSNorm(cfg.d_model, cfg.rmsnorm_eps)
        self.self_attn = GroupedQueryAttention(cfg, causal=True)
        self.self_attn_gate = nn.Parameter(torch.zeros(1))
        self.encoder_attn_layer_norm = ZCRMSNorm(cfg.d_model, cfg.rmsnorm_eps)
        self.encoder_attn = GroupedQueryAttention(cfg)
        self.cross_attn_gate = nn.Parameter(torch.zeros(1))
        self.cross_attention_rope = cfg.cross_attention_rope

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        target_valid: torch.Tensor,
        source_valid: torch.Tensor,
        target_positions: torch.Tensor,
        source_positions: torch.Tensor,
    ) -> torch.Tensor:
        normalized = self.input_layernorm(x)
        update = self.self_attn(
            normalized,
            normalized,
            key_valid=target_valid,
            query_positions=target_positions,
            key_positions=target_positions,
        )
        x = x + torch.sigmoid(self.self_attn_gate) * update
        update = self.encoder_attn(
            self.encoder_attn_layer_norm(x),
            memory,
            key_valid=source_valid,
            query_positions=target_positions,
            key_positions=source_positions,
            causal=False,
            use_rope=self.cross_attention_rope,
        )
        return x + torch.sigmoid(self.cross_attn_gate) * update


class NeedleishModel(nn.Module):
    def __init__(self, cfg: NeedleConfig):
        super().__init__()
        self.cfg = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.encoder = nn.ModuleList(EncoderBlock(cfg) for _ in range(cfg.encoder_layers))
        self.decoder = nn.ModuleList(DecoderBlock(cfg) for _ in range(cfg.decoder_layers))
        self.encoder_final_norm = ZCRMSNorm(cfg.d_model, cfg.rmsnorm_eps)
        self.decoder_final_norm = ZCRMSNorm(cfg.d_model, cfg.rmsnorm_eps)
        self.apply(self._init)

    @staticmethod
    def _init(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def encode(self, source_ids: torch.Tensor, source_valid: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(source_ids.shape[1], device=source_ids.device)
        x = self.token_embedding(source_ids) * self.cfg.embedding_scale
        for block in self.encoder:
            x = block(x, source_valid, positions)
        return self.encoder_final_norm(x)

    def decode_hidden(
        self,
        decoder_input_ids: torch.Tensor,
        memory: torch.Tensor,
        source_valid: torch.Tensor,
        target_valid: torch.Tensor,
    ) -> torch.Tensor:
        source_positions = torch.arange(memory.shape[1], device=memory.device)
        target_positions = torch.arange(decoder_input_ids.shape[1], device=memory.device)
        x = self.token_embedding(decoder_input_ids) * self.cfg.embedding_scale
        for block in self.decoder:
            x = block(x, memory, target_valid, source_valid, target_positions, source_positions)
        return self.decoder_final_norm(x)

    def decode(
        self,
        decoder_input_ids: torch.Tensor,
        memory: torch.Tensor,
        source_valid: torch.Tensor,
        target_valid: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.decode_hidden(decoder_input_ids, memory, source_valid, target_valid)
        return F.linear(hidden, self.token_embedding.weight)

    def forward(
        self,
        source_ids: torch.Tensor,
        source_valid: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        target_valid: torch.Tensor,
    ) -> torch.Tensor:
        return self.decode(decoder_input_ids, self.encode(source_ids, source_valid), source_valid, target_valid)

    def parameter_breakdown(self) -> dict[str, int]:
        groups = {
            "shared_embedding": self.token_embedding,
            "encoder": self.encoder,
            "decoder": self.decoder,
            "final_norms": nn.ModuleList([self.encoder_final_norm, self.decoder_final_norm]),
        }
        return {name: sum(parameter.numel() for parameter in module.parameters()) for name, module in groups.items()}

    def n_params(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def attention_stats(self) -> dict[str, float]:
        modules = [module for module in self.modules() if isinstance(module, GroupedQueryAttention)]
        return {
            "attention/logit_std": sum(module.last_logit_std for module in modules) / max(len(modules), 1),
            "attention/q_norm": 1.0,
            "attention/k_norm": 1.0,
        }


def load_public_checkpoint(model: NeedleishModel, path: str | Path) -> None:
    """Load the pinned Hugging Face safetensors release into NeedleishModel."""
    from safetensors.torch import load_file

    if model.cfg != NeedleConfig.public_checkpoint():
        raise ValueError("Public Needle weights require NeedleConfig.public_checkpoint()")

    released = load_file(str(path), device="cpu")
    mapped: dict[str, torch.Tensor] = {}
    for name, value in released.items():
        if name == "lm_head.weight":  # Exact duplicate of the tied embedding.
            continue
        if name == "model.embed_tokens.weight":
            target = "token_embedding.weight"
        elif name == "model.encoder.final_norm.weight":
            target = "encoder_final_norm.scale"
        elif name == "model.decoder.norm.weight":
            target = "decoder_final_norm.scale"
        elif name.startswith("model.encoder.layers."):
            target = name.removeprefix("model.encoder.layers.")
            target = f"encoder.{target}"
        elif name.startswith("model.decoder.layers."):
            target = name.removeprefix("model.decoder.layers.")
            target = f"decoder.{target}"
        else:
            raise ValueError(f"Unknown public Needle tensor: {name}")
        if target.endswith("_norm.weight") or target.endswith("layernorm.weight"):
            target = target.removesuffix("weight") + "scale"
        mapped[target] = value

    missing, unexpected = model.load_state_dict(mapped, strict=False)
    if missing or unexpected:
        raise ValueError(f"Checkpoint mapping mismatch: missing={missing}, unexpected={unexpected}")
