from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class ModelConfig:
    model_name: str = "attn_pg_23m_v1"
    vocab_size: int = 32_000
    d_model: int = 384
    n_heads: int = 6
    encoder_layers: int = 6
    decoder_layers: int = 6
    transformer_ffn: bool = False
    rmsnorm_eps: float = 1.0e-5
    rope_theta: float = 10_000.0
    dropout: float = 0.10
    source_length: int = 512
    target_length: int = 64
    num_token_types: int = 2
    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2
    cls_id: int = 3
    sep_id: int = 4
    question_id: int = 5
    context_id: int = 6
    answer_start_mode: str = "context"

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.transformer_ffn:
            raise ValueError("The main experiment must have transformer_ffn=false")
        if self.head_dim % 2:
            raise ValueError("RoPE requires an even head dimension")
        if self.answer_start_mode not in {"context", "global"}:
            raise ValueError("answer_start_mode must be context or global")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ModelConfig":
        import yaml

        with Path(path).open() as handle:
            values = yaml.safe_load(handle)
        model = values.get("model", values)
        tokenizer = values.get("tokenizer", {})
        sequence = values.get("sequence", {})
        architecture = values.get("architecture", {})
        return cls(
            model_name=values.get("model_name", cls.model_name),
            vocab_size=tokenizer.get("vocab_size", cls.vocab_size),
            d_model=architecture.get("d_model", cls.d_model),
            n_heads=architecture.get("n_heads", cls.n_heads),
            encoder_layers=architecture.get("encoder_layers", cls.encoder_layers),
            decoder_layers=architecture.get("decoder_layers", cls.decoder_layers),
            transformer_ffn=architecture.get("transformer_ffn", cls.transformer_ffn),
            rmsnorm_eps=architecture.get("rmsnorm_eps", cls.rmsnorm_eps),
            rope_theta=architecture.get("rope_theta", cls.rope_theta),
            dropout=architecture.get("dropout", cls.dropout),
            source_length=sequence.get("source_length_initial", cls.source_length),
            target_length=sequence.get("target_length", cls.target_length),
            answer_start_mode=architecture.get("answer_start_mode", cls.answer_start_mode),
        )
