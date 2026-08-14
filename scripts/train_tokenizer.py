#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grounded_qa.synthetic import generate_synthetic
from grounded_qa.tokenizer import train_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/tokenizer.json")
    parser.add_argument("--train-n", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vocab-size", type=int, default=32_000)
    args = parser.parse_args()

    rows = generate_synthetic(args.train_n, args.seed, "train")
    info = train_tokenizer(
        (f"{row['question']} {row['context']}" for row in rows),
        args.output,
        args.vocab_size,
    )
    print(f"saved {info.path}")
    print(f"special_ids: {info}")


if __name__ == "__main__":
    main()
