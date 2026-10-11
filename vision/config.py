"""Config loading, seeding and device selection."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REQUIRED_KEYS = (
    "seed", "classes", "positive_class", "camera", "data", "preprocess",
    "augment", "model", "train", "evaluate", "inference", "server",
)


def load_config(path: str | Path = "config.yaml") -> dict[str, Any]:
    """Load and sanity-check the YAML config."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    if missing:
        raise KeyError(f"config {path} is missing keys: {missing}")
    if cfg["positive_class"] not in cfg["classes"]:
        raise ValueError("positive_class must be one of classes")
    n_blocks = len(cfg["model"]["channels"])
    if not 4 <= n_blocks <= 6:
        raise ValueError(f"model.channels must have 4-6 entries, got {n_blocks}")
    return cfg


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch for reproducible runs."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(name: str = "auto") -> "torch.device":
    """Return cuda if requested/available, else cpu."""
    import torch

    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device=cuda requested but CUDA is not available")
    return torch.device(name)
