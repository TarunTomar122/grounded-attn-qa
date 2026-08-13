from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import Dataset

from .tokenizer import TokenizerInfo


@dataclass
class EncodedExample:
    source_ids: list[int]
    token_type_ids: list[int]
    source_valid: list[bool]
    context_mask: list[bool]
    decoder_input_ids: list[int]
    target_ids: list[int]
    target_valid: list[bool]
    gold_copy_positions: list[int]
    answerable: bool
    row: dict[str, Any]


def _one(tokenizer, token: str) -> int:
    ids = tokenizer.encode(token, add_special_tokens=False).ids
    if len(ids) != 1:
        raise ValueError(f"Expected one token for {token}, got {ids}")
    return ids[0]


def encode_row(
    row: dict[str, Any],
    tokenizer,
    info: TokenizerInfo,
    *,
    max_source_length: int,
    max_target_length: int,
) -> EncodedExample | None:
    question_ids = tokenizer.encode(row["question"], add_special_tokens=False).ids
    # Byte-level BPE represents a word differently at string-start than after
    # whitespace. Give source context and answer targets the same boundary so
    # the pointer can copy the first answer token.
    metadata = row.get("metadata", {})
    context_prefix = "" if metadata.get("tokenizer_boundary") == "raw" else " "
    context_ids = tokenizer.encode(context_prefix + row["context"], add_special_tokens=False).ids
    source_ids = [info.cls_id, info.question_id, *question_ids, info.sep_id, info.context_id, *context_ids, info.eos_id]
    token_type_ids = [0] * (3 + len(question_ids)) + [1] * (1 + len(context_ids)) + [0]
    context_mask = [False] * (4 + len(question_ids)) + [True] * len(context_ids) + [False]
    if len(source_ids) > max_source_length:
        return None

    context_encoding = tokenizer.encode(context_prefix + row["context"], add_special_tokens=False)
    answer_ids = tokenizer.encode(" " + row.get("answer", ""), add_special_tokens=False).ids
    gold_copy_positions: list[int] | None = None
    if "window_answer_start" in metadata and "window_answer_end" in metadata:
        start = int(metadata["window_answer_start"])
        end = int(metadata["window_answer_end"])
        source_positions = [
            index
            for index, (token_start, token_end) in enumerate(context_encoding.offsets)
            if token_end > start and token_start < end + (0 if metadata.get("tokenizer_boundary") == "raw" else 1)
        ]
        aligned_ids = [context_encoding.ids[index] for index in source_positions]
        if not source_positions or tokenizer.decode(aligned_ids).strip() != row["answer"].strip():
            raise ValueError(f"answer span does not align to source tokens: {row['answer']!r}")
        answer_ids = aligned_ids
        gold_copy_positions = [4 + len(question_ids) + position for position in source_positions] + [-1]
    answer_ids = answer_ids[: max_target_length - 1]
    decoder_input_ids = [info.bos_id, *answer_ids]
    target_ids = [*answer_ids, info.eos_id]
    if not row.get("answerable", False):
        decoder_input_ids = [info.bos_id]
        target_ids = [info.eos_id]
        gold_copy_positions = [-1]
    elif gold_copy_positions is None:
        gold_copy_positions = [-1] * len(target_ids)
        context_start = 4 + len(question_ids)
        # The full evidence sentence can have a different BPE boundary after
        # a newline; the answer span has the stable whitespace boundary used
        # by the decoder target. Procedural rows make answer values unique.
        for offset in range(len(context_ids) - len(answer_ids) + 1):
            if context_ids[offset : offset + len(answer_ids)] == answer_ids:
                start = context_start + offset
                gold_copy_positions = list(range(start, start + len(answer_ids))) + [-1]
                break
        if answer_ids and gold_copy_positions[0] < 0:
            raise ValueError(f"answer is not copyable from context: {row['answer']!r}")
    else:
        gold_copy_positions = gold_copy_positions[: len(target_ids) - 1] + [-1]
    return EncodedExample(
        source_ids=source_ids,
        token_type_ids=token_type_ids,
        source_valid=[True] * len(source_ids),
        context_mask=context_mask,
        decoder_input_ids=decoder_input_ids,
        target_ids=target_ids,
        target_valid=[bool(row.get("answerable", False))] * len(target_ids),
        gold_copy_positions=gold_copy_positions,
        answerable=bool(row.get("answerable", False)),
        row=row,
    )


class GroundedDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], tokenizer, info: TokenizerInfo, *, source_length: int, target_length: int):
        self.examples = [
            encoded
            for row in rows
            if (encoded := encode_row(row, tokenizer, info, max_source_length=source_length, max_target_length=target_length))
            is not None
        ]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> EncodedExample:
        return self.examples[index]


def collate_examples(batch: list[EncodedExample], pad_id: int) -> dict[str, torch.Tensor | list[dict[str, Any]]]:
    source_length = max(len(row.source_ids) for row in batch)
    target_length = max(len(row.decoder_input_ids) for row in batch)

    def long_field(name: str, pad: int) -> torch.Tensor:
        length = target_length if name in {"decoder_input_ids", "target_ids", "gold_copy_positions"} else source_length
        output = torch.full((len(batch), length), pad, dtype=torch.long)
        for i, row in enumerate(batch):
            values = getattr(row, name)
            output[i, : len(values)] = torch.tensor(values, dtype=torch.long)
        return output

    def bool_field(name: str, length: int) -> torch.Tensor:
        output = torch.zeros((len(batch), length), dtype=torch.bool)
        for i, row in enumerate(batch):
            values = getattr(row, name)
            output[i, : len(values)] = torch.tensor(values, dtype=torch.bool)
        return output

    return {
        "source_ids": long_field("source_ids", pad_id),
        "token_type_ids": long_field("token_type_ids", 0),
        "source_valid": bool_field("source_valid", source_length),
        "context_mask": bool_field("context_mask", source_length),
        "decoder_input_ids": long_field("decoder_input_ids", pad_id),
        "target_ids": long_field("target_ids", pad_id),
        "gold_copy_positions": long_field("gold_copy_positions", -1),
        "target_valid": bool_field("target_valid", target_length),
        "answerable": torch.tensor([row.answerable for row in batch], dtype=torch.bool),
        "rows": [row.row for row in batch],
    }
