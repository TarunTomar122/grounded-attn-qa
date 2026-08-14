"""Grounded attention-only pointer-generator experiment."""

from .config import ModelConfig
from .model import GroundedPointerGenerator, GroundedOutput

__all__ = ["GroundedOutput", "GroundedPointerGenerator", "ModelConfig"]
