from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .model import GroundedOutput


@dataclass
class LossOutput:
    total: torch.Tensor
    sequence: torch.Tensor
    answerability: torch.Tensor
    pointer_position: torch.Tensor


def sequence_nll(
    output: GroundedOutput,
    source_ids: torch.Tensor,
    target_ids: torch.Tensor,
    target_valid: torch.Tensor,
    *,
    copy_only: bool = False,
    eos_id: int | None = None,
) -> torch.Tensor:
    same_token = source_ids[:, None, :].eq(target_ids[:, :, None])
    copy_prob = (output.copy_position_probs * same_token).sum(dim=-1)
    if copy_only:
        if eos_id is None:
            raise ValueError("copy-only loss needs eos_id")
        final_prob = torch.where(
            target_ids.eq(eos_id),
            output.stop_probability,
            (1 - output.stop_probability) * copy_prob,
        )
    else:
        log_vocab = output.vocab_logits.log_softmax(dim=-1)
        vocab_prob = log_vocab.gather(-1, target_ids[..., None]).squeeze(-1).exp()
        final_prob = output.p_gen * vocab_prob + (1 - output.p_gen) * copy_prob
    loss = -final_prob.clamp_min(1.0e-8).log()
    return (loss * target_valid).sum() / target_valid.sum().clamp_min(1)


def grounded_loss(
    output: GroundedOutput,
    source_ids: torch.Tensor,
    target_ids: torch.Tensor,
    target_valid: torch.Tensor,
    answerable: torch.Tensor,
    lambda_answerability: float = 1.0,
    *,
    copy_only: bool = False,
    eos_id: int | None = None,
    gold_copy_positions: torch.Tensor | None = None,
    lambda_pointer_position: float = 0.0,
) -> LossOutput:
    positive_mask = target_valid & answerable[:, None]
    sequence = sequence_nll(
        output,
        source_ids,
        target_ids,
        positive_mask,
        copy_only=copy_only,
        eos_id=eos_id,
    )
    answerability = F.binary_cross_entropy_with_logits(
        output.answerability_logits,
        answerable.float(),
    )
    if lambda_pointer_position and gold_copy_positions is None:
        raise ValueError("pointer-position loss needs gold_copy_positions")
    if gold_copy_positions is None:
        pointer_position = sequence.new_zeros(())
    else:
        pointer_mask = target_valid & answerable[:, None] & gold_copy_positions.ge(0)
        safe_positions = gold_copy_positions.clamp_min(0)
        pointer_probability = output.copy_position_probs.gather(-1, safe_positions[..., None]).squeeze(-1)
        pointer_loss = -pointer_probability.clamp_min(1.0e-8).log()
        pointer_position = (pointer_loss * pointer_mask).sum() / pointer_mask.sum().clamp_min(1)
    total = sequence + lambda_answerability * answerability + lambda_pointer_position * pointer_position
    return LossOutput(total=total, sequence=sequence, answerability=answerability, pointer_position=pointer_position)
