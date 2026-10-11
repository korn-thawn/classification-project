"""Dataset scanning, session-based splitting and the torch Dataset.

Splitting is by capture session, never by individual image, so near-duplicate
frames from one session cannot leak between train/val/test.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from vision.augment import Augmenter
from vision.preprocess import normalize, prepare_uint8

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}
Item = tuple[Path, int]


def scan_sessions(raw_dir: str | Path, classes: list[str]) -> dict[str, list[Item]]:
    """Return {session_name: [(image_path, label_idx), ...]} sorted by session name."""
    raw_dir = Path(raw_dir)
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"data dir not found: {raw_dir}")
    sessions: dict[str, list[Item]] = {}
    for sdir in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
        items: list[Item] = []
        for idx, name in enumerate(classes):
            cdir = sdir / name
            if cdir.is_dir():
                items += [(p, idx) for p in sorted(cdir.iterdir()) if p.suffix.lower() in IMAGE_EXTS]
        if items:
            sessions[sdir.name] = items
    return sessions


def split_by_session(sessions: dict[str, list[Item]], val_sessions: list[str],
                     test_sessions: list[str]) -> dict[str, list[str]]:
    """Assign whole sessions to train/val/test.

    If val/test lists are empty, split chronologically by session name:
    newest -> test, second newest -> val, the rest -> train.
    """
    names = sorted(sessions)
    unknown = [s for s in val_sessions + test_sessions if s not in sessions]
    if unknown:
        raise ValueError(f"sessions in config not found in data: {unknown}")
    if not val_sessions and not test_sessions:
        if len(names) < 3:
            raise ValueError(f"need at least 3 sessions for train/val/test, found {len(names)}: {names}")
        test_sessions, val_sessions = [names[-1]], [names[-2]]
    elif not val_sessions or not test_sessions:
        raise ValueError("set both data.val_sessions and data.test_sessions, or leave both empty")
    train = [s for s in names if s not in val_sessions and s not in test_sessions]
    split = {"train": train, "val": list(val_sessions), "test": list(test_sessions)}
    assert not set(split["train"]) & set(split["val"]), "train/val sessions overlap"
    assert not set(split["train"]) & set(split["test"]), "train/test sessions overlap"
    assert not set(split["val"]) & set(split["test"]), "val/test sessions overlap"
    if not train:
        raise ValueError("no sessions left for training")
    return split


def collect(sessions: dict[str, list[Item]], names: list[str]) -> list[Item]:
    """Flatten the items of the given sessions."""
    return [it for s in names for it in sessions[s]]


def class_counts(items: list[Item], num_classes: int) -> np.ndarray:
    """Number of images per class."""
    return np.bincount([y for _, y in items], minlength=num_classes)


def summarize(split: dict[str, list[str]], sessions: dict[str, list[Item]],
              classes: list[str]) -> str:
    """Human-readable split summary; warns when a split lacks a class."""
    lines = []
    for part, names in split.items():
        counts = class_counts(collect(sessions, names), len(classes))
        c = ", ".join(f"{n}={k}" for n, k in zip(classes, counts))
        lines.append(f"{part:5s}: {len(names)} session(s) {names} -> {c}")
        if (counts == 0).any():
            lines.append(f"  WARNING: {part} has no images of some class")
    return "\n".join(lines)


def load_image(path: Path) -> np.ndarray:
    """Read an image as BGR uint8 (works with non-ASCII Windows paths)."""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"cannot read image {path}")
    return img


try:
    from torch.utils.data import Dataset as _TorchDataset
except ImportError:  # allows scan/split helpers without torch installed
    _TorchDataset = object  # type: ignore[assignment,misc]


class OkNgDataset(_TorchDataset):
    """Returns (float32 CHW tensor, label). Augments only if augmenter is given."""

    def __init__(self, items: list[Item], cfg: dict[str, Any],
                 augmenter: Augmenter | None = None, cache: bool = True) -> None:
        self.items = items
        self.cfg = cfg
        self.augmenter = augmenter
        self.cache = [prepare_uint8(load_image(p), cfg) for p, _ in items] if cache else None

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        import torch

        path, label = self.items[i]
        img = self.cache[i] if self.cache is not None else prepare_uint8(load_image(path), self.cfg)
        if self.augmenter is not None:
            img = self.augmenter(img)
        p = self.cfg["preprocess"]
        return torch.from_numpy(normalize(img, p["mean"], p["std"])), label
