from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.utils.data import Dataset

from .needle_tokenizer import NeedleTokenizer, SPECIAL_TOKENS


@dataclass
class EncodedSynthExample:
    source_ids: list[int]
    target_ids: list[int]
    decoder_input_ids: list[int]
    reasoning_mask: list[bool]
    answer_mask: list[bool]
    query: str
    context: str
    gold_reasoning: str
    gold_answer: str
    source_url: str
    exercise: str
    source_tokens: int
    target_tokens: int
    reasoning_tokens: int
    answer_tokens: int
    answer_source_overlap: float
    row_id: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def source_bucket(source_url: str) -> int:
    digest = hashlib.sha256(source_url.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % 1000


def split_for_source(source_url: str) -> str:
    return "validation" if source_bucket(source_url) < 10 else "train"


def encode_synth_row(
    row: dict[str, Any],
    tokenizer: NeedleTokenizer,
    *,
    source_max_length: int = 512,
    target_max_length: int = 256,
    answer_max_length: int = 96,
) -> EncodedSynthExample | None:
    query = row.get("query")
    context = row.get("query_seed_text")
    reasoning = row.get("synthetic_reasoning")
    answer = row.get("synthetic_answer")
    if not all(isinstance(value, str) and value.strip() for value in (query, context, reasoning, answer)):
        return None
    source_ids = tokenizer.encode_source(query, context)
    if len(source_ids) > source_max_length:
        return None
    reasoning_ids = tokenizer.encode(reasoning)
    answer_ids = tokenizer.encode(answer)
    answer_prefix = [SPECIAL_TOKENS["<ANSWER>"], *answer_ids, 1]
    reasoning_prefix = [SPECIAL_TOKENS["<REASONING>"]]
    if len(answer_ids) > answer_max_length or len(reasoning_prefix) + len(answer_prefix) > target_max_length:
        return None
    reasoning_budget = target_max_length - len(reasoning_prefix) - len(answer_prefix)
    target_ids = reasoning_prefix + reasoning_ids[:reasoning_budget] + answer_prefix
    reasoning_start = 1
    reasoning_end = reasoning_start + min(len(reasoning_ids), reasoning_budget)
    answer_start = reasoning_end
    answer_end = len(target_ids)
    reasoning_mask = [False] * len(target_ids)
    answer_mask = [False] * len(target_ids)
    reasoning_mask[0:reasoning_end] = [True] * reasoning_end
    answer_mask[answer_start:answer_end] = [True] * (answer_end - answer_start)
    context_ids = tokenizer.encode(context)
    overlap = sum(token_id in set(context_ids) for token_id in answer_ids) / max(len(answer_ids), 1)
    source_url = str(row.get("query_seed_url") or row.get("synth_id") or "missing-source")
    return EncodedSynthExample(
        source_ids=source_ids,
        target_ids=target_ids,
        decoder_input_ids=[2, *target_ids[:-1]],
        reasoning_mask=reasoning_mask,
        answer_mask=answer_mask,
        query=query,
        context=context,
        gold_reasoning=reasoning,
        gold_answer=answer,
        source_url=source_url,
        exercise=str(row.get("exercise") or "unknown"),
        source_tokens=len(source_ids),
        target_tokens=len(target_ids),
        reasoning_tokens=max(0, reasoning_end - 1),
        answer_tokens=len(answer_ids),
        answer_source_overlap=overlap,
        row_id=str(row.get("synth_id") or ""),
    )


class EncodedJsonlDataset(Dataset):
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.offsets: list[int] = []
        with self.path.open("rb") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if line.strip():
                    self.offsets.append(offset)
        self._handle = None
        self._handle_pid = None

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self._handle is None or self._handle_pid != os.getpid():
            # Offsets are byte offsets, so keep the worker handle binary too.
            self._handle = self.path.open("rb")
            self._handle_pid = os.getpid()
        self._handle.seek(self.offsets[index])
        return json.loads(self._handle.readline())


def collate_synth(batch: list[dict[str, Any]], pad_id: int = 0) -> dict[str, torch.Tensor | list[dict[str, Any]]]:
    source_length = max(len(row["source_ids"]) for row in batch)
    target_length = max(len(row["target_ids"]) for row in batch)

    def ids(name: str, length: int, fill: int) -> torch.Tensor:
        output = torch.full((len(batch), length), fill, dtype=torch.long)
        for index, row in enumerate(batch):
            values = row[name]
            output[index, : len(values)] = torch.tensor(values, dtype=torch.long)
        return output

    def mask(name: str, length: int) -> torch.Tensor:
        output = torch.zeros((len(batch), length), dtype=torch.bool)
        for index, row in enumerate(batch):
            values = row[name]
            output[index, : len(values)] = torch.tensor(values, dtype=torch.bool)
        return output

    source_ids = ids("source_ids", source_length, pad_id)
    target_ids = ids("target_ids", target_length, pad_id)
    return {
        "source_ids": source_ids,
        "source_valid": source_ids.ne(pad_id),
        "decoder_input_ids": ids("decoder_input_ids", target_length, pad_id),
        "target_ids": target_ids,
        "target_valid": mask("target_ids", target_length),
        "reasoning_mask": mask("reasoning_mask", target_length),
        "answer_mask": mask("answer_mask", target_length),
        "metadata": batch,
    }


def write_encoded_jsonl(rows: Iterable[EncodedSynthExample], path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")
            count += 1
    return count
