from __future__ import annotations

import torch

from .model import GroundedPointerGenerator


@torch.no_grad()
def generate(
    model: GroundedPointerGenerator,
    source_ids: torch.Tensor,
    token_type_ids: torch.Tensor,
    source_valid: torch.Tensor,
    context_mask: torch.Tensor,
    *,
    bos_id: int,
    eos_id: int,
    max_new_tokens: int,
    copy_only: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    memory = model.encode(source_ids, token_type_ids, source_valid)
    decoder_ids = torch.full(
        (source_ids.shape[0], 1),
        bos_id,
        dtype=torch.long,
        device=source_ids.device,
    )
    p_answerable = torch.sigmoid(model.answerability(memory[:, 0]).squeeze(-1))
    pointer_steps: list[torch.Tensor] = []
    finished = torch.zeros(source_ids.shape[0], dtype=torch.bool, device=source_ids.device)
    for _ in range(max_new_tokens):
        target_valid = torch.ones_like(decoder_ids, dtype=torch.bool)
        output = model.decode(
            decoder_ids,
            memory,
            source_ids,
            context_mask,
            source_valid,
            target_valid,
        )
        distribution = output.final_distribution(source_ids, copy_only=copy_only, eos_id=eos_id)
        next_id = distribution[:, -1].argmax(dim=-1)
        next_id = torch.where(finished, torch.full_like(next_id, eos_id), next_id)
        pointer_steps.append(output.copy_position_probs[:, -1])
        decoder_ids = torch.cat((decoder_ids, next_id[:, None]), dim=1)
        finished |= next_id.eq(eos_id)
        if finished.all():
            break
    return decoder_ids, p_answerable, pointer_steps
