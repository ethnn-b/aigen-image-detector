"""Small shared helpers: device choice and seeding.

Kept tiny on purpose. The interesting code lives in metrics, data, and train; this
is just the plumbing they all reach for.
"""

from __future__ import annotations

import random

import numpy as np
import torch


def get_device(prefer: str | None = None) -> torch.device:
    """Pick the best available device.

    Order is MPS (Apple Silicon), then CUDA, then CPU. Pass `prefer` to force one,
    which is what the tests do to stay on CPU.
    """
    if prefer:
        return torch.device(prefer)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    """Seed python, numpy, and torch so a run is repeatable."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
