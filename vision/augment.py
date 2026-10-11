"""Data augmentation written with NumPy + OpenCV only.

Every function takes a BGR uint8 image and returns a BGR uint8 image of the
same shape. Random functions take an explicit np.random.Generator so results
are reproducible from a seed.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import cv2
import numpy as np


def hflip(img: np.ndarray) -> np.ndarray:
    """Mirror left-right."""
    return np.ascontiguousarray(img[:, ::-1])


def vflip(img: np.ndarray) -> np.ndarray:
    """Mirror top-bottom."""
    return np.ascontiguousarray(img[::-1])


def affine(img: np.ndarray, angle_deg: float, tx: float, ty: float, scale: float) -> np.ndarray:
    """Rotate about the centre, scale, then shift by (tx, ty) pixels; borders are reflected."""
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, scale)
    m[0, 2] += tx
    m[1, 2] += ty
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


def brightness(img: np.ndarray, delta: float) -> np.ndarray:
    """Add delta (in 0-255 units) to every pixel."""
    return np.clip(img.astype(np.float32) + delta, 0, 255).astype(np.uint8)


def contrast(img: np.ndarray, factor: float) -> np.ndarray:
    """Scale distance from the image mean by factor."""
    f = img.astype(np.float32)
    mean = f.mean()
    return np.clip((f - mean) * factor + mean, 0, 255).astype(np.uint8)


def gamma(img: np.ndarray, g: float) -> np.ndarray:
    """Apply gamma curve out = 255 * (in/255) ** g via a lookup table."""
    lut = np.clip(255.0 * (np.arange(256) / 255.0) ** g, 0, 255).astype(np.uint8)
    return cv2.LUT(img, lut)


def gaussian_noise(img: np.ndarray, std: float, rng: np.random.Generator) -> np.ndarray:
    """Add zero-mean gaussian noise with the given std (0-255 units)."""
    if std <= 0:
        return img.copy()
    noise = rng.normal(0.0, std, size=img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def gaussian_blur(img: np.ndarray, ksize: int) -> np.ndarray:
    """Gaussian blur with an odd kernel size."""
    if ksize % 2 == 0 or ksize < 1:
        raise ValueError(f"blur kernel size must be odd and positive, got {ksize}")
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


def _uniform(rng: np.random.Generator, r: Sequence[float]) -> float:
    return float(rng.uniform(r[0], r[1]))


class Augmenter:
    """Apply the augmentations enabled in config['augment'] in a fixed order."""

    ORDER = ("hflip", "vflip", "affine", "brightness", "contrast", "gamma", "noise", "blur")

    def __init__(self, cfg_augment: dict[str, Any], seed: int | None = None) -> None:
        self.cfg = cfg_augment
        self.rng = np.random.default_rng(seed)

    def _on(self, name: str) -> bool:
        c = self.cfg.get(name)
        return bool(c and c.get("enabled", False) and self.rng.random() < c.get("p", 1.0))

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if not self.cfg.get("enabled", False):
            return img
        out = img
        rng = self.rng
        for name in self.ORDER:
            if not self._on(name):
                continue
            c = self.cfg[name]
            if name == "hflip":
                out = hflip(out)
            elif name == "vflip":
                out = vflip(out)
            elif name == "affine":
                h, w = out.shape[:2]
                t = c["max_translate_frac"]
                out = affine(
                    out,
                    angle_deg=_uniform(rng, (-c["max_rotate_deg"], c["max_rotate_deg"])),
                    tx=_uniform(rng, (-t, t)) * w,
                    ty=_uniform(rng, (-t, t)) * h,
                    scale=_uniform(rng, c["scale_range"]),
                )
            elif name == "brightness":
                out = brightness(out, _uniform(rng, (-c["max_delta"], c["max_delta"])))
            elif name == "contrast":
                out = contrast(out, _uniform(rng, c["range"]))
            elif name == "gamma":
                out = gamma(out, _uniform(rng, c["range"]))
            elif name == "noise":
                out = gaussian_noise(out, _uniform(rng, (0.0, c["max_std"])), rng)
            elif name == "blur":
                out = gaussian_blur(out, int(rng.choice(c["kernel_sizes"])))
        return out
