from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import nn

from .needleish import NeedleConfig, NeedleishModel
from .pointer import PointerGenerator


def _masked_mean(memory: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (memory * mask[..., None]).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1)


def answerability_interaction_features(
    memory: torch.Tensor, source_valid: torch.Tensor, context_mask: torch.Tensor
) -> torch.Tensor:
    """Linear-readout features that retain both question and context summaries."""
    question_mask = source_valid & ~context_mask
    question = _masked_mean(memory, question_mask)
    context = _masked_mean(memory, context_mask)
    return torch.cat((question, context, question * context, (question - context).abs()), dim=-1)


def candidate_span_features(
    memory: torch.Tensor, source_valid: torch.Tensor, question_mask: torch.Tensor, candidate_mask: torch.Tensor
) -> torch.Tensor:
    """Read the jointly encoded question against the exact proposed source span."""
    question = _masked_mean(memory, source_valid & question_mask)
    candidate = _masked_mean(memory, source_valid & candidate_mask)
    return torch.cat((question, candidate, question * candidate, (question - candidate).abs()), dim=-1)


@dataclass
class NeedlePointerOutput:
    vocab_logits: torch.Tensor
    copy_position_probs: torch.Tensor
    p_gen: torch.Tensor
    answerability_logits: torch.Tensor | None = None
    evidence_position_logits: torch.Tensor | None = None

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
        # Heads are experimental and may change shape; the pretrained reader and
        # pointer must still match exactly when resuming an older checkpoint.
        expected = self.state_dict()
        compatible = {name: value for name, value in state.items() if name in expected and expected[name].shape == value.shape}
        missing, unexpected = self.load_state_dict(compatible, strict=False)
        optional_heads = ("pointer.", "answerability.", "evidence.", "no_answer_logit")
        if unexpected or any(not name.startswith(optional_heads) for name in missing):
            raise ValueError(f"Backbone checkpoint mismatch: missing={missing}, unexpected={unexpected}")


class NeedleAnswerablePointerModel(NeedlePointerModel):
    """Pointer-generator with a question-conditioned answer-span/no-answer head."""

    def __init__(self, cfg: NeedleConfig):
        super().__init__(cfg)
        self.evidence = nn.Linear(cfg.d_model, 1)
        nn.init.zeros_(self.evidence.weight)
        nn.init.zeros_(self.evidence.bias)
        # Start roughly neutral for a typical 256-token context.
        self.no_answer_logit = nn.Parameter(torch.tensor(math.log(256.0)))

    def forward(
        self,
        source_ids: torch.Tensor,
        source_valid: torch.Tensor,
        context_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        target_valid: torch.Tensor,
    ) -> NeedlePointerOutput:
        memory = self.encode(source_ids, source_valid)
        output = self.decode_pointer(
            decoder_input_ids,
            memory,
            source_ids,
            source_valid,
            context_mask,
            target_valid,
        )
        evidence_position_logits = self.evidence_position_logits(memory, context_mask)
        answerability_logits = self.classify_answerability(memory, source_valid, context_mask, evidence_position_logits)
        return NeedlePointerOutput(
            output.vocab_logits,
            output.copy_position_probs,
            output.p_gen,
            answerability_logits,
            evidence_position_logits,
        )

    def evidence_position_logits(self, memory: torch.Tensor, context_mask: torch.Tensor) -> torch.Tensor:
        """Scores a context start position, with index zero reserved for no-answer."""
        positions = self.evidence(memory).squeeze(-1).masked_fill(~context_mask, float("-inf"))
        return torch.cat((self.no_answer_logit.expand(len(memory), 1), positions), dim=1)

    def classify_answerability(
        self,
        memory: torch.Tensor,
        source_valid: torch.Tensor,
        context_mask: torch.Tensor,
        evidence_position_logits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del source_valid  # Context already excludes source padding.
        logits = evidence_position_logits if evidence_position_logits is not None else self.evidence_position_logits(memory, context_mask)
        return logits[:, 1:].logsumexp(dim=-1) - logits[:, 0]


def evidence_start_targets(gold_copy_positions: torch.Tensor, answerable: torch.Tensor) -> torch.Tensor:
    """Index zero is no-answer; source positions are shifted by one."""
    starts = gold_copy_positions[:, 0]
    if (answerable & starts.lt(0)).any():
        raise ValueError("answerable rows require a gold first copy position")
    return torch.where(answerable, starts + 1, torch.zeros_like(starts))


def evidence_start_loss(output: NeedlePointerOutput, gold_copy_positions: torch.Tensor, answerable: torch.Tensor) -> torch.Tensor:
    """Supervise the exact evidence start, or the no-answer class for negatives."""
    if output.evidence_position_logits is None:
        raise ValueError("evidence_start_loss requires NeedleAnswerablePointerModel output")
    return F.cross_entropy(output.evidence_position_logits.float(), evidence_start_targets(gold_copy_positions, answerable))


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
    selected_logits = logits[target_valid]
    z = selected_logits.logsumexp(dim=-1).square().mean() if selected_logits.numel() else logits.sum() * 0

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
