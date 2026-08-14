from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .needleish import NeedleConfig, NeedleishModel
from .pointer import PointerGenerator


@dataclass
class NeedlePointerOutput:
    vocab_logits: torch.Tensor
    copy_position_probs: torch.Tensor
    p_gen: torch.Tensor

    def final_distribution(self, source_ids: torch.Tensor) -> torch.Tensor:
        copy = PointerGenerator.copy_distribution(
            self.copy_position_probs,
            source_ids,
            self.vocab_logits.shape[-1],
        )
        return self.p_gen[..., None] * self.vocab_logits.softmax(dim=-1) + (1 - self.p_gen[..., None]) * copy


class NeedlePointerModel(NeedleishModel):
    """Public Needle backbone plus the project's existing pointer-generator head."""

    def __init__(self, cfg: NeedleConfig):
        super().__init__(cfg)
        self.pointer = PointerGenerator(cfg.d_model)
        self.pointer.apply(self._init)
        nn.init.zeros_(self.pointer.gate.bias)

    def forward(
        self,
        source_ids: torch.Tensor,
        source_valid: torch.Tensor,
        context_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        target_valid: torch.Tensor,
    ) -> NeedlePointerOutput:
        memory = self.encode(source_ids, source_valid)
        return self.decode_pointer(
            decoder_input_ids,
            memory,
            source_ids,
            source_valid,
            context_mask,
            target_valid,
        )

    def decode_pointer(
        self,
        decoder_input_ids: torch.Tensor,
        memory: torch.Tensor,
        source_ids: torch.Tensor,
        source_valid: torch.Tensor,
        context_mask: torch.Tensor,
        target_valid: torch.Tensor,
    ) -> NeedlePointerOutput:
        hidden = self.decode_hidden(decoder_input_ids, memory, source_valid, target_valid)
        vocab_logits, copy_probs, p_gen, _ = self.pointer(
            hidden,
            memory,
            source_ids,
            context_mask & source_valid,
            self.token_embedding(decoder_input_ids),
            self.token_embedding.weight,
        )
        return NeedlePointerOutput(vocab_logits, copy_probs, p_gen)

    def load_backbone_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        missing, unexpected = self.load_state_dict(state, strict=False)
        if unexpected or any(not name.startswith("pointer.") for name in missing):
            raise ValueError(f"Backbone checkpoint mismatch: missing={missing}, unexpected={unexpected}")


@dataclass
class NeedlePointerLoss:
    total: torch.Tensor
    sequence: torch.Tensor
    z: torch.Tensor
    pointer_position: torch.Tensor
    pointer_accuracy: torch.Tensor
    mean_gold_pointer_probability: torch.Tensor
    mean_p_gen: torch.Tensor


def pointer_loss(
    output: NeedlePointerOutput,
    source_ids: torch.Tensor,
    target_ids: torch.Tensor,
    target_valid: torch.Tensor,
    gold_copy_positions: torch.Tensor,
    *,
    z_weight: float = 1.0e-4,
    pointer_weight: float = 1.0,
    first_token_weight: float = 1.0,
    eos_token_weight: float = 1.0,
    eos_id: int = 1,
) -> NeedlePointerLoss:
    logits = output.vocab_logits.float()
    vocab_prob = logits.log_softmax(dim=-1).gather(-1, target_ids[..., None]).squeeze(-1).exp()
    same_token = source_ids[:, None, :].eq(target_ids[:, :, None])
    copy_prob = (output.copy_position_probs.float() * same_token).sum(dim=-1)
    final_prob = output.p_gen.float() * vocab_prob + (1 - output.p_gen.float()) * copy_prob
    per_token = -final_prob.clamp_min(1.0e-8).log()
    first = torch.arange(target_ids.shape[1], device=target_ids.device).eq(0)[None]
    weights = 1 + first * (first_token_weight - 1) + target_ids.eq(eos_id) * (eos_token_weight - 1)
    weights = weights * target_valid
    sequence = (per_token * weights).sum() / weights.sum().clamp_min(1)
    z = logits[target_valid].logsumexp(dim=-1).square().mean()

    supervised = target_valid & gold_copy_positions.ge(0)
    safe_positions = gold_copy_positions.clamp_min(0)
    supervised_prob = output.copy_position_probs.gather(-1, safe_positions[..., None]).squeeze(-1)
    position_nll = -supervised_prob.clamp_min(1.0e-8).log()
    pointer_position = (position_nll * supervised).sum() / supervised.sum().clamp_min(1)
    pointer_accuracy = (
        output.copy_position_probs.argmax(dim=-1).eq(gold_copy_positions) & supervised
    ).sum() / supervised.sum().clamp_min(1)
    mean_gold_pointer_probability = (supervised_prob * supervised).sum() / supervised.sum().clamp_min(1)
    mean_p_gen = (output.p_gen * target_valid).sum() / target_valid.sum().clamp_min(1)
    return NeedlePointerLoss(
        sequence + z_weight * z + pointer_weight * pointer_position,
        sequence,
        z,
        pointer_position,
        pointer_accuracy,
        mean_gold_pointer_probability,
        mean_p_gen,
    )
