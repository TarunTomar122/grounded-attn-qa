from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import nn

from .needleish import NeedleConfig, NeedleishModel


@dataclass
class SpanNullOutput:
    start_logits: torch.Tensor
    end_logits: torch.Tensor
    memory: torch.Tensor


def _masked_mean(memory: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(memory.dtype)[..., None]
    return (memory * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


class NeedleSpanNullModel(nn.Module):
    """Encoder-only extractive QA head with NULL as a real competing class."""

    def __init__(self, cfg: NeedleConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or NeedleConfig.public_checkpoint()
        self.backbone = NeedleishModel(self.cfg)
        self.start_head = nn.Linear(self.cfg.d_model, 1)
        self.end_head = nn.Linear(self.cfg.d_model, 1)
        self.null_start_key = nn.Parameter(torch.empty(self.cfg.d_model))
        self.null_end_key = nn.Parameter(torch.empty(self.cfg.d_model))
        self.null_start_bias = nn.Parameter(torch.zeros(()))
        self.null_end_bias = nn.Parameter(torch.zeros(()))
        nn.init.normal_(self.null_start_key, mean=0.0, std=0.02)
        nn.init.normal_(self.null_end_key, mean=0.0, std=0.02)

    def forward(
        self,
        source_ids: torch.Tensor,
        source_valid: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> SpanNullOutput:
        memory = self.backbone.encode(source_ids, source_valid)
        valid_context = source_valid & context_mask
        source_start = self.start_head(memory).squeeze(-1)
        source_end = self.end_head(memory).squeeze(-1)
        source_start = source_start.masked_fill(~valid_context, float("-inf"))
        source_end = source_end.masked_fill(~valid_context, float("-inf"))

        question_summary = _masked_mean(memory, source_valid & ~context_mask)
        scale = math.sqrt(self.cfg.d_model)
        null_start = self.null_start_bias + question_summary @ self.null_start_key / scale
        null_end = self.null_end_bias + question_summary @ self.null_end_key / scale
        return SpanNullOutput(
            torch.cat((null_start[:, None], source_start), dim=1),
            torch.cat((null_end[:, None], source_end), dim=1),
            memory,
        )


def span_null_loss(
    output: SpanNullOutput,
    gold_start: torch.Tensor,
    gold_end: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    start_loss = F.cross_entropy(output.start_logits.float(), gold_start)
    end_loss = F.cross_entropy(output.end_logits.float(), gold_end)
    return start_loss + end_loss, start_loss, end_loss


@torch.no_grad()
def best_spans(
    output: SpanNullOutput,
    *,
    max_answer_length: int = 30,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return best source start/end, its score, and NULL-minus-span margin."""
    starts = output.start_logits[:, 1:].float()
    ends = output.end_logits[:, 1:].float()
    batch, length = starts.shape
    best_start = torch.zeros(batch, dtype=torch.long, device=starts.device)
    best_end = torch.zeros(batch, dtype=torch.long, device=starts.device)
    best_score = torch.full((batch,), -torch.inf, device=starts.device)
    for start in range(length):
        end = min(length, start + max_answer_length)
        scores = starts[:, start, None] + ends[:, start:end]
        values, offsets = scores.max(dim=1)
        better = values > best_score
        best_score = torch.where(better, values, best_score)
        best_start = torch.where(better, torch.full_like(best_start, start + 1), best_start)
        best_end = torch.where(better, start + 1 + offsets, best_end)
    null_score = output.start_logits[:, 0].float() + output.end_logits[:, 0].float()
    return best_start, best_end, best_score, null_score - best_score


def threshold_predictions(
    output: SpanNullOutput,
    *,
    threshold: float = 0.0,
    max_answer_length: int = 30,
) -> tuple[torch.Tensor, torch.Tensor]:
    start, end, _, margin = best_spans(output, max_answer_length=max_answer_length)
    null = margin >= threshold
    start = torch.where(null, torch.zeros_like(start), start)
    end = torch.where(null, torch.zeros_like(end), end)
    return start, end
