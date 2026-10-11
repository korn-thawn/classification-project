"""Webcam capture tool: press o = save OK, n = save NG, q = quit.

Frames are saved uncropped to data/raw/{session}/{OK|NG}/{timestamp}.png so
the ROI can be changed later without re-capturing.
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from vision.config import load_config

KEY_TO_CLASS = {ord("o"): "OK", ord("n"): "NG"}
CLASS_COLORS = {"OK": (0, 200, 0), "NG": (0, 0, 255)}


def open_camera(index: int, width: int, height: int) -> cv2.VideoCapture:
    """Open the webcam (DirectShow on Windows opens much faster)."""
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open camera index {index}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def draw_overlay(frame: np.ndarray, roi: list[int] | None, counts: dict[str, int],
                 session: str, flash: tuple[str, float] | None) -> np.ndarray:
    """Draw ROI box, per-class counts and key help on a copy of the frame."""
    view = frame.copy()
    if roi is not None:
        x, y, w, h = roi
        cv2.rectangle(view, (x, y), (x + w, y + h), (255, 200, 0), 2)
    lines = [
        f"session: {session}",
        f"OK: {counts['OK']}   NG: {counts['NG']}",
        "[o] save OK   [n] save NG   [q] quit",
    ]
    for i, text in enumerate(lines):
        org = (10, 30 + 30 * i)
        cv2.putText(view, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(view, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    if flash is not None and time.monotonic() - flash[1] < 0.6:
        color = CLASS_COLORS[flash[0]]
        cv2.rectangle(view, (0, 0), (view.shape[1] - 1, view.shape[0] - 1), color, 12)
        cv2.putText(view, f"saved {flash[0]}", (10, view.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3, cv2.LINE_AA)
    return view


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--check", action="store_true",
                        help="open the camera, grab one frame, print its shape and exit")
    args = parser.parse_args()
    cfg = load_config(args.config)
    cam = cfg["camera"]
    roi = cfg["preprocess"]["roi"]

    cap = open_camera(cam["index"], cam["width"], cam["height"])
    try:
        if args.check:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("camera opened but returned no frame")
            print(f"camera {cam['index']} OK, frame shape = {frame.shape}")
            return

        session = datetime.now().strftime("%Y%m%d-%H%M%S")
        session_dir = Path(cfg["data"]["raw_dir"]) / session
        counts = {"OK": 0, "NG": 0}
        flash: tuple[str, float] | None = None
        print(f"session {session} -> {session_dir}")

        while True:
            ok, frame = cap.read()
            if not ok:
                print("frame grab failed, stopping")
                break
            cv2.imshow("capture (o=OK, n=NG, q=quit)", draw_overlay(frame, roi, counts, session, flash))
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key in KEY_TO_CLASS:
                label = KEY_TO_CLASS[key]
                out_dir = session_dir / label
                out_dir.mkdir(parents=True, exist_ok=True)
                path = out_dir / f"{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}.png"
                cv2.imwrite(str(path), frame)
                counts[label] += 1
                flash = (label, time.monotonic())
        print(f"saved OK={counts['OK']} NG={counts['NG']} in {session_dir}")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
