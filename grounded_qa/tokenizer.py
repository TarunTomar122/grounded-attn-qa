from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers


SPECIAL_TOKENS = ["<PAD>", "<BOS>", "<EOS>", "<CLS>", "<SEP>", "<Q>", "<CTX>", "<UNK>"]


@dataclass(frozen=True)
class TokenizerInfo:
    path: Path
    pad_id: int
    bos_id: int
    eos_id: int
    cls_id: int
    sep_id: int
    question_id: int
    context_id: int


def train_tokenizer(texts: Iterable[str], output_path: str | Path, vocab_size: int = 32_000) -> TokenizerInfo:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer(models.BPE(unk_token="<UNK>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)
    tokenizer.save(str(output_path))
    return tokenizer_info(output_path)


def load_tokenizer(path: str | Path) -> tuple[Tokenizer, TokenizerInfo]:
    path = Path(path)
    return Tokenizer.from_file(str(path)), tokenizer_info(path)


def tokenizer_info(path: str | Path) -> TokenizerInfo:
    tokenizer = Tokenizer.from_file(str(path))

    def token_id(token: str) -> int:
        value = tokenizer.token_to_id(token)
        if value is None:
            raise ValueError(f"Tokenizer is missing special token {token}")
        return value

    return TokenizerInfo(
        path=Path(path),
        pad_id=token_id("<PAD>"),
        bos_id=token_id("<BOS>"),
        eos_id=token_id("<EOS>"),
        cls_id=token_id("<CLS>"),
        sep_id=token_id("<SEP>"),
        question_id=token_id("<Q>"),
        context_id=token_id("<CTX>"),
    )
