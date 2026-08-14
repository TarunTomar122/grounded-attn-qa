#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grounded_qa.config import ModelConfig
from grounded_qa.losses import grounded_loss
from grounded_qa.model import GroundedPointerGenerator


def run_trial(batch_size: int, steps: int, precision: str, device: torch.device) -> dict[str, float | str]:
    cfg = ModelConfig()
    model = GroundedPointerGenerator(cfg).to(device).train()
    source = torch.randint(0, cfg.vocab_size, (batch_size, cfg.source_length), device=device)
    types = torch.zeros_like(source)
    valid = torch.ones_like(source, dtype=torch.bool)
    context = torch.zeros_like(valid)
    context[:, cfg.source_length // 3 :] = True
    decoder = torch.randint(0, cfg.vocab_size, (batch_size, 32), device=device)
    target = torch.randint(0, cfg.vocab_size, decoder.shape, device=device)
    target_valid = torch.ones_like(decoder, dtype=torch.bool)
    answerable = torch.ones(batch_size, dtype=torch.bool, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    started = time.perf_counter()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        if precision == "fp16":
            precision_context = torch.autocast(device_type=device.type, dtype=torch.float16)
        elif precision == "bf16":
            precision_context = torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        else:
            precision_context = contextlib.nullcontext()
        with precision_context:
            output = model(source, types, valid, context, decoder, target_valid)
            losses = grounded_loss(
                output,
                source,
                target,
                target_valid,
                answerable,
                0.0,
                copy_only=True,
                eos_id=cfg.eos_id,
            )
        losses.total.backward()
        optimizer.step()
    elapsed = time.perf_counter() - started
    tokens = steps * batch_size * (cfg.source_length + decoder.shape[1])
    result: dict[str, float | str] = {"device": str(device), "batch_size": batch_size, "precision": precision, "steps": steps, "tokens_per_sec": tokens / elapsed, "step_time_ms": elapsed * 1000 / steps, "loss": losses.total.item()}
    if device.type == "mps":
        result.update({"mps_tensor_gb": torch.mps.current_allocated_memory() / 1024**3, "mps_driver_gb": torch.mps.driver_allocated_memory() / 1024**3, "mps_recommended_gb": torch.mps.recommended_max_memory() / 1024**3})
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--batch-sizes", default="2,4,8,16")
    parser.add_argument("--precisions", default="fp32,fp16,bf16")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS is unavailable on this machine")
    for precision in args.precisions.split(","):
        for batch_size in (int(value) for value in args.batch_sizes.split(",")):
            try:
                print(run_trial(batch_size, args.steps, precision, device), flush=True)
            except RuntimeError as exc:
                print({"batch_size": batch_size, "precision": precision, "error": str(exc)}, flush=True)


if __name__ == "__main__":
    main()
