from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import torch

from .config import ModelConfig
from .model import GroundedPointerGenerator


def save_checkpoint(
    path: str | Path,
    model: GroundedPointerGenerator,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any,
    *,
    step: int,
    tokens_seen: int,
    phase: str,
    config: ModelConfig,
    tokenizer_path: str,
    seed: int,
    wandb_run_id: str | None = None,
    threshold: float | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "step": step,
        "tokens_seen": tokens_seen,
        "phase": phase,
        "config": config.to_dict(),
        "tokenizer_path": tokenizer_path,
        "seed": seed,
        "wandb_run_id": wandb_run_id,
        "threshold": threshold,
        "python_state": random.getstate(),
        "torch_state": torch.get_rng_state().cpu(),
    }
    if torch.cuda.is_available():
        state["cuda_state"] = [value.cpu() for value in torch.cuda.get_rng_state_all()]
    torch.save(state, path)


def load_checkpoint(
    path: str | Path,
    model: GroundedPointerGenerator,
    optimizer=None,
    scheduler=None,
    device="cpu",
    *,
    strict: bool = True,
) -> dict:
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"], strict=strict)
    if optimizer is not None and state.get("optimizer") is not None:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and state.get("scheduler") is not None:
        scheduler.load_state_dict(state["scheduler"])
    if state.get("python_state") is not None:
        random.setstate(state["python_state"])
    if state.get("torch_state") is not None:
        torch.set_rng_state(state["torch_state"].cpu())
    if torch.cuda.is_available() and state.get("cuda_state") is not None:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda_state"]])
    return state
