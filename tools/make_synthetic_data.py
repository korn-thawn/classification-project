"""Generate a synthetic OK/NG dataset to test the pipeline without a camera.

Draws a flange-like part (disc + centre hole + 4 bolt holes) on a textured
background. NG parts get one defect: scratch, edge chip, missing bolt hole or
stain. Each session has its own lighting to mimic real session-to-session
variation. Output layout matches capture.py: {out}/{session}/{OK|NG}/*.png

Also writes {out}/config.yaml = ./config.yaml with raw_dir and an ROI around
the part, so: python train.py --config data/synthetic/config.yaml
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np
import yaml

W, H = 640, 480
DEFECTS = ("scratch", "chip", "missing_hole", "stain")
# square crop that always contains the part (centre jitter +-25/20 px, radius <= 165 px)
ROI = [130, 50, 380, 380]


def draw_part(rng: np.random.Generator, light: float, defect: str | None) -> np.ndarray:
    """Render one 640x480 BGR image of the part, optionally with a defect."""
    bg = rng.normal(70, 6, size=(H, W, 1)).astype(np.float32)
    img = np.repeat(bg, 3, axis=2)
    cx = W // 2 + int(rng.integers(-25, 26))
    cy = H // 2 + int(rng.integers(-20, 21))
    r = int(rng.integers(150, 165))
    rot = rng.uniform(0, 2 * math.pi)
    body = (175, 180, 185)

    cv2.circle(img, (cx, cy), r, body, -1, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), r, (120, 120, 125), 3, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), r // 3, (40, 40, 40), -1, cv2.LINE_AA)

    missing = int(rng.integers(0, 4)) if defect == "missing_hole" else -1
    for k in range(4):
        if k == missing:
            continue
        a = rot + k * math.pi / 2
        hx, hy = int(cx + 0.68 * r * math.cos(a)), int(cy + 0.68 * r * math.sin(a))
        cv2.circle(img, (hx, hy), 14, (40, 40, 40), -1, cv2.LINE_AA)

    if defect == "scratch":
        a = rng.uniform(0, 2 * math.pi)
        d = rng.uniform(0.4, 0.8) * r
        p1 = (int(cx + d * math.cos(a)), int(cy + d * math.sin(a)))
        length = rng.uniform(50, 110)
        b = a + rng.uniform(1.2, 1.9)
        p2 = (int(p1[0] + length * math.cos(b)), int(p1[1] + length * math.sin(b)))
        cv2.line(img, p1, p2, (90, 90, 95), int(rng.integers(2, 4)), cv2.LINE_AA)
    elif defect == "chip":
        a = rng.uniform(0, 2 * math.pi)
        px, py = int(cx + r * math.cos(a)), int(cy + r * math.sin(a))
        cv2.circle(img, (px, py), int(rng.integers(18, 30)), (70, 70, 70), -1, cv2.LINE_AA)
    elif defect == "stain":
        a = rng.uniform(0, 2 * math.pi)
        d = rng.uniform(0.45, 0.85) * r
        sx, sy = int(cx + d * math.cos(a)), int(cy + d * math.sin(a))
        axes = (int(rng.integers(12, 22)), int(rng.integers(8, 16)))
        cv2.ellipse(img, (sx, sy), axes, rng.uniform(0, 180), 0, 360, (95, 120, 140), -1, cv2.LINE_AA)

    # session lighting: global gain + a horizontal gradient
    grad = np.linspace(1.0 - 0.15 * light, 1.0 + 0.15 * light, W, dtype=np.float32)[None, :, None]
    img = img * (0.85 + 0.3 * rng.random()) * grad
    img += rng.normal(0, 3, size=img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/synthetic")
    parser.add_argument("--sessions", type=int, default=5)
    parser.add_argument("--ok", type=int, default=150, help="OK images per session")
    parser.add_argument("--ng", type=int, default=60, help="NG images per session")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--base-config", default="config.yaml")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    for s in range(args.sessions):
        session = f"20260101-{9 + s:02d}0000"
        light = rng.uniform(-1, 1)
        for label, n in (("OK", args.ok), ("NG", args.ng)):
            d = out / session / label
            d.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                defect = DEFECTS[i % len(DEFECTS)] if label == "NG" else None
                cv2.imwrite(str(d / f"{session}-{i:04d}.png"), draw_part(rng, light, defect))
        print(f"{session}: OK={args.ok} NG={args.ng}")

    with open(args.base_config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["data"]["raw_dir"] = out.as_posix()
    cfg["preprocess"]["roi"] = ROI
    with open(out / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    print(f"done -> {out} (config with ROI {ROI}: {out / 'config.yaml'})")


if __name__ == "__main__":
    main()
