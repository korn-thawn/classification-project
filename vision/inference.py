"""Checkpoint loading and scoring shared by evaluate.py and server.py.

Always uses the config stored inside the checkpoint for preprocessing, so
inference matches what the model was trained on even if config.yaml changed.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vision.config import resolve_device
from vision.model import build_model
from vision.preprocess import preprocess


def resolve_checkpoint(cfg: dict[str, Any], override: str | None = None) -> Path:
    """Checkpoint path from CLI override, config inference.checkpoint, or runs/LATEST."""
    path = override or cfg["inference"].get("checkpoint")
    if path:
        return Path(path)
    latest = Path(cfg["train"]["out_dir"]) / "LATEST"
    if not latest.is_file():
        raise FileNotFoundError(f"no checkpoint configured and {latest} not found; run train.py first")
    return Path(cfg["train"]["out_dir"]) / latest.read_text(encoding="utf-8").strip() / "best.pt"


class Predictor:
    """Loaded model + its training config; scores = probability of the positive (NG) class."""

    def __init__(self, checkpoint: str | Path, device: str = "auto", default_threshold: float = 0.5) -> None:
        self.checkpoint = Path(checkpoint)
        self.device = resolve_device(device)
        ckpt = torch.load(self.checkpoint, map_location=self.device, weights_only=True)
        self.cfg: dict[str, Any] = ckpt["config"]
        self.classes: list[str] = ckpt["classes"]
        self.model_version: str = ckpt["model_version"]
        self.split: dict[str, list[str]] = ckpt["split"]
        self.pos_class: str = self.cfg["positive_class"]
        self.pos_idx = self.classes.index(self.pos_class)
        self.neg_class = next(c for c in self.classes if c != self.pos_class)
        self.model = build_model(self.cfg).to(self.device).eval()
        self.model.load_state_dict(ckpt["model_state"])
        self.threshold = self._load_threshold(default_threshold)

    @property
    def run_dir(self) -> Path:
        return self.checkpoint.parent

    def _load_threshold(self, default: float) -> float:
        path = self.run_dir / "threshold.json"
        if path.is_file():
            return float(json.loads(path.read_text(encoding="utf-8"))["threshold"])
        return float(default)

    @torch.inference_mode()
    def score_batch(self, images: list[np.ndarray], batch_size: int = 64) -> np.ndarray:
        """P(positive class) for a list of BGR uint8 images."""
        out = []
        for i in range(0, len(images), batch_size):
            x = np.stack([preprocess(img, self.cfg) for img in images[i:i + batch_size]])
            logits = self.model(torch.from_numpy(x).to(self.device))
            out.append(torch.softmax(logits, dim=1)[:, self.pos_idx].float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)

    @torch.inference_mode()
    def warmup(self) -> None:
        """Run one dummy forward pass (skips preprocessing, so ROI size does not matter)."""
        s = int(self.cfg["preprocess"]["image_size"])
        self.model(torch.zeros(1, 3, s, s, device=self.device))

    def score(self, image: np.ndarray) -> float:
        """P(positive class) for a single BGR uint8 image."""
        return float(self.score_batch([image])[0])

    def decide(self, score: float, threshold: float | None = None) -> str:
        """Class name for a score: positive class if score >= threshold."""
        t = self.threshold if threshold is None else threshold
        return self.pos_class if score >= t else self.neg_class
