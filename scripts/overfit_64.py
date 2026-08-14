#!/usr/bin/env python3
"""Mandatory implementation check: fit 64 synthetic rows repeatedly."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grounded_qa.config import ModelConfig
from grounded_qa.data import GroundedDataset, collate_examples
from grounded_qa.losses import grounded_loss
from grounded_qa.model import GroundedPointerGenerator
from grounded_qa.synthetic import generate_synthetic
from grounded_qa.tokenizer import load_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/attn_pg_23m.yaml")
    parser.add_argument("--tokenizer", default="artifacts/tokenizer.json")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    tokenizer, info = load_tokenizer(args.tokenizer)
    cfg = ModelConfig.from_yaml(args.config)
    cfg.vocab_size = tokenizer.get_vocab_size()
    cfg.pad_id, cfg.bos_id, cfg.eos_id = info.pad_id, info.bos_id, info.eos_id
    cfg.cls_id, cfg.sep_id = info.cls_id, info.sep_id
    cfg.question_id, cfg.context_id = info.question_id, info.context_id
    model = GroundedPointerGenerator(cfg).to(device).train()
    model.answerability.requires_grad_(False)
    rows = generate_synthetic(64, 42, "train")
    dataset = GroundedDataset(rows, tokenizer, info, source_length=cfg.source_length, target_length=cfg.target_length)
    if len(dataset) != 64:
        raise SystemExit(f"64-row overfit setup lost examples: {len(dataset)} remain")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda x: collate_examples(x, cfg.pad_id))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    iterator = iter(loader)
    for step in range(1, args.steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        output = model(batch["source_ids"], batch["token_type_ids"], batch["source_valid"], batch["context_mask"], batch["decoder_input_ids"], batch["target_valid"])
        losses = grounded_loss(
            output,
            batch["source_ids"],
            batch["target_ids"],
            batch["target_valid"],
            batch["answerable"],
            0.0,
            copy_only=True,
            eos_id=cfg.eos_id,
        )
        losses.total.backward()
        clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == args.steps:
            print(f"step={step} loss={losses.total.item():.4f}", flush=True)

    model.eval()
    correct = total = 0
    with torch.inference_mode():
        for batch in DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=lambda x: collate_examples(x, cfg.pad_id)):
            batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
            output = model(batch["source_ids"], batch["token_type_ids"], batch["source_valid"], batch["context_mask"], batch["decoder_input_ids"], batch["target_valid"])
            predicted = output.final_distribution(
                batch["source_ids"], copy_only=True, eos_id=cfg.eos_id
            ).argmax(-1)
            correct += int(((predicted == batch["target_ids"]) & batch["target_valid"]).sum())
            total += int(batch["target_valid"].sum())
    accuracy = correct / max(total, 1)
    print(f"token_accuracy={accuracy:.4f} ({correct}/{total})")
    if accuracy < 0.98:
        raise SystemExit("64-example overfit gate failed")


if __name__ == "__main__":
    main()
