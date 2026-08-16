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
    memory: torch.Tensor
    decoder_hidden: torch.Tensor


class NeedleFullSpanNullModel(nn.Module):
    """Full pretrained Needle encoder-decoder source pointer with first-class NULL.

    The decoder runs two learned decisions from the public Needle decoder-start
    token. Each decision scores NULL against every valid context position. No
    vocabulary distribution is used, so every non-NULL prediction is grounded
    in the supplied context.
    """

    def __init__(self, cfg: NeedleConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or NeedleConfig.public_checkpoint()
        self.backbone = NeedleishModel(self.cfg)
        self.pointer = PointerGenerator(self.cfg.d_model)
        self.pointer.apply(NeedleishModel._init)
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

        query = self.pointer.pointer_q(decoder_hidden)
        key = self.pointer.pointer_k(memory)
        source_logits = torch.einsum("btd,bsd->bts", query, key) / math.sqrt(self.cfg.d_model)
        valid_context = source_valid & context_mask
        source_logits = source_logits.masked_fill(~valid_context[:, None, :], float("-inf"))

        null_start = self.null_start_bias + decoder_hidden[:, 0] @ self.null_start_key / math.sqrt(self.cfg.d_model)
        null_end = self.null_end_bias + decoder_hidden[:, 1] @ self.null_end_key / math.sqrt(self.cfg.d_model)
        return FullSpanNullOutput(
            start_logits=torch.cat((null_start[:, None], source_logits[:, 0]), dim=1),
            end_logits=torch.cat((null_end[:, None], source_logits[:, 1]), dim=1),
            memory=memory,
            decoder_hidden=decoder_hidden,
        )
