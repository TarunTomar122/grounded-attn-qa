from __future__ import annotations

from pathlib import Path

import sentencepiece as spm


SPECIAL_TOKENS = {
    "<QUERY>": 8192,
    "<CONTEXT>": 8193,
    "<REASONING>": 8194,
    "<ANSWER>": 8195,
}


class NeedleTokenizer:
    """SentencePiece tokenizer with four appended atomic training markers."""

    def __init__(self, model_path: str | Path):
        self.path = Path(model_path)
        self.sp = spm.SentencePieceProcessor(model_file=str(self.path))
        if self.sp.vocab_size() != 8192:
            raise ValueError(f"Needle tokenizer must have 8192 base pieces, got {self.sp.vocab_size()}")

    @property
    def vocab_size(self) -> int:
        return 8192 + len(SPECIAL_TOKENS)

    def encode(self, text: str) -> list[int]:
        return list(self.sp.encode(text, out_type=int))

    def decode(self, ids: list[int]) -> str:
        pieces: list[str] = []
        base: list[int] = []
        reverse = {value: key for key, value in SPECIAL_TOKENS.items()}
        for token_id in ids:
            if token_id in reverse:
                if base:
                    pieces.append(self.sp.decode(base))
                    base = []
                pieces.append(reverse[token_id])
            else:
                base.append(int(token_id))
        if base:
            pieces.append(self.sp.decode(base))
        return "".join(pieces)

    def encode_source(self, query: str, context: str) -> list[int]:
        return [SPECIAL_TOKENS["<QUERY>"], *self.encode(query), SPECIAL_TOKENS["<CONTEXT>"], *self.encode(context)]

    def encode_target(self, reasoning: str, answer: str, eos_id: int = 1) -> tuple[list[int], dict[str, int]]:
        reasoning_ids = self.encode(reasoning)
        answer_ids = self.encode(answer)
        target = [SPECIAL_TOKENS["<REASONING>"], *reasoning_ids, SPECIAL_TOKENS["<ANSWER>"], *answer_ids, eos_id]
        return target, {"reasoning_tokens": len(reasoning_ids), "answer_tokens": len(answer_ids)}
