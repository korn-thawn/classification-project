"""Preprocessing shared by training and inference.

Input is always a BGR uint8 image (what cv2.imread / cv2.imdecode return).
Training:  prepare_uint8 -> augment -> normalize
Inference: preprocess (= prepare_uint8 -> normalize)
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import cv2
import numpy as np


def crop_roi(img: np.ndarray, roi: Sequence[int] | None) -> np.ndarray:
    """Crop [x, y, w, h] from img; None returns the image unchanged."""
    if roi is None:
        return img
    x, y, w, h = (int(v) for v in roi)
    H, W = img.shape[:2]
    if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > W or y + h > H:
        raise ValueError(f"ROI {list(roi)} is outside the image of size {W}x{H} (WxH)")
    return img[y:y + h, x:x + w]


def resize(img: np.ndarray, size: int) -> np.ndarray:
    """Resize to size x size (INTER_AREA when shrinking, INTER_LINEAR when enlarging)."""
    shrinking = img.shape[0] > size or img.shape[1] > size
    interp = cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR
    return cv2.resize(img, (size, size), interpolation=interp)


def ensure_bgr(img: np.ndarray) -> np.ndarray:
    """Convert grayscale or BGRA input to 3-channel BGR."""
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def prepare_uint8(img: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    """Crop ROI and resize; output is BGR uint8 (size, size, 3)."""
    p = cfg["preprocess"]
    img = ensure_bgr(img)
    return resize(crop_roi(img, p["roi"]), int(p["image_size"]))


def normalize(img: np.ndarray, mean: Sequence[float], std: Sequence[float]) -> np.ndarray:
    """BGR uint8 HWC -> RGB float32 CHW, scaled to [0,1] then (x - mean) / std."""
    rgb = img[:, :, ::-1].astype(np.float32) / 255.0
    rgb = (rgb - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    return np.ascontiguousarray(rgb.transpose(2, 0, 1))


def preprocess(img: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    """Full inference preprocessing: crop, resize, normalize -> float32 (3, S, S)."""
    p = cfg["preprocess"]
    return normalize(prepare_uint8(img, cfg), p["mean"], p["std"])
