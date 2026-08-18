from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn

from .needleish import NeedleConfig, NeedleishModel
from .pointer import PointerGenerator


@dataclass
class FullSpanNullOutput:
    start_logits: torch.Tensor
    end_logits: torch.Tensor
    independent_start_logits: torch.Tensor
    independent_end_logits: torch.Tensor
    memory: torch.Tensor
    decoder_hidden: torch.Tensor


class NeedleFullSpanNullModel(nn.Module):
    """Full pretrained Needle encoder-decoder source pointer with first-class NULL.

    The decoder runs two learned decisions from the public Needle decoder-start
    token. Each joint decision scores NULL against every valid context position.
    A second source-only pointer pair learns answer localization without NULL
    competition, while all heads share the same pretrained backbone.
    """

    def __init__(self, cfg: NeedleConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or NeedleConfig.public_checkpoint()
        self.backbone = NeedleishModel(self.cfg)
        self.start_pointer = PointerGenerator(self.cfg.d_model)
        self.end_pointer = PointerGenerator(self.cfg.d_model)
        self.independent_start_pointer = PointerGenerator(self.cfg.d_model)
        self.independent_end_pointer = PointerGenerator(self.cfg.d_model)
        self.start_pointer.apply(NeedleishModel._init)
        self.end_pointer.apply(NeedleishModel._init)
        self.independent_start_pointer.apply(NeedleishModel._init)
        self.independent_end_pointer.apply(NeedleishModel._init)
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
    ) -> FullSpanNullOutput:
        memory = self.backbone.encode(source_ids, source_valid)
        decoder_input_ids = torch.full(
            (source_ids.shape[0], 2),
            self.cfg.eos_id,
            dtype=torch.long,
            device=source_ids.device,
        )
        target_valid = torch.ones_like(decoder_input_ids, dtype=torch.bool)
        decoder_hidden = self.backbone.decode_hidden(
            decoder_input_ids,
            memory,
            source_valid,
            target_valid,
        )

        valid_context = source_valid & context_mask
        start_query = self.start_pointer.pointer_q(decoder_hidden[:, 0:1])
        start_key = self.start_pointer.pointer_k(memory)
        start_logits = torch.einsum("btd,bsd->bts", start_query, start_key) / math.sqrt(self.cfg.d_model)
        start_logits = start_logits.masked_fill(~valid_context[:, None, :], float("-inf"))

        end_query = self.end_pointer.pointer_q(decoder_hidden[:, 1:2])
        end_key = self.end_pointer.pointer_k(memory)
        end_logits = torch.einsum("btd,bsd->bts", end_query, end_key) / math.sqrt(self.cfg.d_model)
        end_logits = end_logits.masked_fill(~valid_context[:, None, :], float("-inf"))

        independent_start_query = self.independent_start_pointer.pointer_q(decoder_hidden[:, 0:1])
        independent_start_key = self.independent_start_pointer.pointer_k(memory)
        independent_start_logits = torch.einsum(
            "btd,bsd->bts", independent_start_query, independent_start_key
        ) / math.sqrt(self.cfg.d_model)
        independent_start_logits = independent_start_logits.masked_fill(
            ~valid_context[:, None, :], float("-inf")
        )

        independent_end_query = self.independent_end_pointer.pointer_q(decoder_hidden[:, 1:2])
        independent_end_key = self.independent_end_pointer.pointer_k(memory)
        independent_end_logits = torch.einsum(
            "btd,bsd->bts", independent_end_query, independent_end_key
        ) / math.sqrt(self.cfg.d_model)
        independent_end_logits = independent_end_logits.masked_fill(
            ~valid_context[:, None, :], float("-inf")
        )

        null_start = self.null_start_bias + decoder_hidden[:, 0] @ self.null_start_key / math.sqrt(self.cfg.d_model)
        null_end = self.null_end_bias + decoder_hidden[:, 1] @ self.null_end_key / math.sqrt(self.cfg.d_model)
        return FullSpanNullOutput(
            start_logits=torch.cat((null_start[:, None], start_logits[:, 0]), dim=1),
            end_logits=torch.cat((null_end[:, None], end_logits[:, 0]), dim=1),
            independent_start_logits=independent_start_logits[:, 0],
            independent_end_logits=independent_end_logits[:, 0],
            memory=memory,
            decoder_hidden=decoder_hidden,
        )


def load_compatible_state_dict(model: nn.Module, state_dict: dict[str, torch.Tensor]) -> bool:
    """Load old checkpoints, which predate the independent pointer pair."""
    incompatible = model.load_state_dict(state_dict, strict=False)
    unexpected = set(incompatible.unexpected_keys)
    unexpected_missing = {
        key
        for key in incompatible.missing_keys
        if not key.startswith(("independent_start_pointer.", "independent_end_pointer."))
    }
    if unexpected or unexpected_missing:
        raise RuntimeError(
            f"incompatible Needle checkpoint: missing={sorted(unexpected_missing)}, "
            f"unexpected={sorted(unexpected)}"
        )
    return bool(incompatible.missing_keys)
