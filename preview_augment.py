"""Save grids of augmented samples so augmentation settings can be checked by eye.

Each output PNG = one source image: the first tile is the preprocessed
original (ROI crop + resize), the rest are random augmentations of it.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from vision.augment import Augmenter
from vision.config import load_config
from vision.dataset import scan_sessions
from vision.preprocess import prepare_uint8


def make_grid(tiles: list[np.ndarray], cols: int, pad: int = 4) -> np.ndarray:
    """Arrange equal-size BGR tiles into a grid with white padding."""
    h, w = tiles[0].shape[:2]
    rows = (len(tiles) + cols - 1) // cols
    grid = np.full((rows * (h + pad) + pad, cols * (w + pad) + pad, 3), 255, np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        y, x = pad + r * (h + pad), pad + c * (w + pad)
        grid[y:y + h, x:x + w] = t
    return grid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--data-dir", help="override data.raw_dir")
    parser.add_argument("--image", action="append", help="specific image path (repeatable)")
    parser.add_argument("--per-class", type=int, default=2, help="source images per class")
    parser.add_argument("--n", type=int, default=15, help="augmented variants per image")
    parser.add_argument("--out", default="outputs/augment_preview")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.image:
        sources = [(Path(p), "img") for p in args.image]
    else:
        raw_dir = args.data_dir or cfg["data"]["raw_dir"]
        items = [it for v in scan_sessions(raw_dir, cfg["classes"]).values() for it in v]
        rng = np.random.default_rng(cfg["seed"])
        sources = []
        for label_idx, name in enumerate(cfg["classes"]):
            paths = [p for p, y in items if y == label_idx]
            for i in rng.permutation(len(paths))[:args.per_class]:
                sources.append((paths[i], name))
    if not sources:
        raise SystemExit("no images found")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    aug = Augmenter(cfg["augment"], seed=cfg["seed"])
    for k, (path, name) in enumerate(sources):
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"skip unreadable {path}")
            continue
        base = prepare_uint8(img, cfg)
        tiles = [base] + [aug(base) for _ in range(args.n)]
        out_path = out_dir / f"{k:02d}_{name}_{path.stem}.png"
        cv2.imwrite(str(out_path), make_grid(tiles, cols=4))
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
