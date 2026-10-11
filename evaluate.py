"""Evaluate a trained checkpoint and pick an NG threshold.

The threshold is chosen on the VAL split (highest threshold whose NG recall
meets evaluate.target_ng_recall, i.e. the fewest false rejects), then the
final numbers are reported on the untouched TEST split.
Outputs go to runs/{model_version}/eval/.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import confusion_matrix, recall_score, roc_auc_score

from vision.config import load_config
from vision.dataset import collect, load_image, scan_sessions
from vision.inference import Predictor, resolve_checkpoint


def pick_threshold(scores: np.ndarray, is_pos: np.ndarray, target_recall: float) -> float:
    """Threshold meeting target NG recall with the fewest false rejects.

    Starts from the highest t with recall(score >= t) >= target_recall (the k-th
    highest NG score), then moves halfway down to the next lower score so the
    threshold does not sit exactly on a sample. Results on these scores are
    unchanged, but unseen NG scored slightly lower are still caught.
    """
    pos_scores = np.sort(scores[is_pos])[::-1]
    if len(pos_scores) == 0:
        raise ValueError("no positive (NG) samples to pick a threshold from")
    k = max(1, math.ceil(target_recall * len(pos_scores)))
    t = float(pos_scores[k - 1])
    below = scores[scores < t]
    return (t + float(below.max())) / 2 if len(below) else t


def metrics_at(scores: np.ndarray, is_pos: np.ndarray, t: float) -> dict[str, Any]:
    """Confusion matrix [[TN, FP], [FN, TP]] (rows = true OK/NG), NG recall, false reject rate."""
    pred = scores >= t
    cm = confusion_matrix(is_pos, pred, labels=[False, True])
    tn, fp, fn, tp = cm.ravel()
    n_ok = tn + fp
    return {
        "threshold": t,
        "confusion_matrix": cm.tolist(),
        "ng_recall": float(recall_score(is_pos, pred, zero_division=0)) if is_pos.any() else None,
        "false_reject_rate": float(fp / n_ok) if n_ok else None,
        "missed_ng": int(fn),
        "false_reject": int(fp),
        "accuracy": float((tp + tn) / len(scores)) if len(scores) else None,
        "n_ok": int(n_ok),
        "n_ng": int(tp + fn),
    }


def sweep(scores: np.ndarray, is_pos: np.ndarray, ts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """NG recall and false reject rate for each threshold in ts."""
    pos, neg = scores[is_pos], scores[~is_pos]
    recall = np.array([(pos >= t).mean() if len(pos) else np.nan for t in ts])
    frr = np.array([(neg >= t).mean() if len(neg) else np.nan for t in ts])
    return recall, frr


def plot_threshold_curve(curves: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
                         target: float, chosen: float, path: Path) -> None:
    """NG recall / false reject rate vs threshold for each split."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    styles = {"val": "--", "test": "-"}
    for name, (ts, rec, frr) in curves.items():
        ax.plot(ts, rec, styles.get(name, "-"), color="tab:red", label=f"NG recall ({name})")
        ax.plot(ts, frr, styles.get(name, "-"), color="tab:blue", label=f"false reject rate ({name})")
    ax.axhline(target, color="gray", lw=1, ls=":", label=f"target NG recall = {target}")
    ax.axvline(chosen, color="black", lw=1.2, label=f"chosen threshold = {chosen:.4f}")
    ax.set_xlabel("threshold on P(NG)")
    ax.set_ylabel("rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(loc="center left", fontsize=8)
    ax.set_title("Threshold selection (picked on val, reported on test)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_confusion(cm: list[list[int]], labels: list[str], title: str, path: Path) -> None:
    """Confusion matrix heatmap with counts."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arr = np.array(cm)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(arr, cmap="Blues")
    for (i, j), v in np.ndenumerate(arr):
        ax.text(j, i, str(v), ha="center", va="center",
                color="white" if v > arr.max() / 2 else "black", fontsize=14)
    ax.set_xticks([0, 1], [f"pred {l}" for l in labels])
    ax.set_yticks([0, 1], [f"true {l}" for l in labels])
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def score_split(pred: Predictor, sessions: dict, names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Scores and is-NG flags for all images of the given sessions."""
    items = collect(sessions, names)
    scores = pred.score_batch([load_image(p) for p, _ in items])
    is_pos = np.array([y == pred.pos_idx for _, y in items], dtype=bool)
    return scores, is_pos


def fmt(m: dict[str, Any]) -> str:
    (tn, fp), (fn, tp) = m["confusion_matrix"]
    rec = "n/a" if m["ng_recall"] is None else f"{m['ng_recall']:.4f}"
    frr = "n/a" if m["false_reject_rate"] is None else f"{m['false_reject_rate']:.4f}"
    return (f"  threshold {m['threshold']:.4f}\n"
            f"                 pred OK  pred NG\n"
            f"    true OK  {tn:9d} {fp:8d}\n"
            f"    true NG  {fn:9d} {tp:8d}\n"
            f"  NG recall {rec}  (missed NG = {m['missed_ng']})\n"
            f"  false reject rate {frr}  (OK rejected = {m['false_reject']})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint", help="path to best.pt (default: inference.checkpoint or runs/LATEST)")
    parser.add_argument("--data-dir", help="override data dir (default: the one used for training)")
    parser.add_argument("--target-recall", type=float, help="override evaluate.target_ng_recall")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save-threshold", action="store_true",
                        help="write the chosen threshold to runs/{version}/threshold.json for server.py")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ckpt_path = resolve_checkpoint(cfg, args.checkpoint)
    pred = Predictor(ckpt_path, device=args.device)
    target = args.target_recall or cfg["evaluate"]["target_ng_recall"]
    labels = [pred.neg_class, pred.pos_class]
    print(f"checkpoint {ckpt_path} (model_version {pred.model_version})")

    data_dir = args.data_dir or pred.cfg["data"]["raw_dir"]
    sessions = scan_sessions(data_dir, pred.classes)
    val_s, val_y = score_split(pred, sessions, pred.split["val"])
    test_s, test_y = score_split(pred, sessions, pred.split["test"])

    chosen = pick_threshold(val_s, val_y, target)
    default_t = float(cfg["inference"]["threshold"])
    results = {
        "model_version": pred.model_version,
        "checkpoint": str(ckpt_path),
        "target_ng_recall": target,
        "chosen_threshold": chosen,
        "split": pred.split,
        "val": {"at_chosen": metrics_at(val_s, val_y, chosen)},
        "test": {
            "at_chosen": metrics_at(test_s, test_y, chosen),
            "at_default": metrics_at(test_s, test_y, default_t),
            "roc_auc": float(roc_auc_score(test_y, test_s)) if 0 < test_y.sum() < len(test_y) else None,
        },
    }

    print(f"\nthreshold picked on VAL for NG recall >= {target}: {chosen:.4f}")
    print(fmt(results["val"]["at_chosen"]))
    print(f"\nTEST @ chosen threshold ({len(test_s)} images, sessions {pred.split['test']}):")
    print(fmt(results["test"]["at_chosen"]))
    print(f"\nTEST @ default threshold {default_t}:")
    print(fmt(results["test"]["at_default"]))
    if results["test"]["roc_auc"] is not None:
        print(f"\nTEST ROC AUC: {results['test']['roc_auc']:.4f}")
    rec = results["test"]["at_chosen"]["ng_recall"]
    if rec is not None and rec < target:
        print(f"\nWARNING: test NG recall {rec:.4f} is below target {target}; collect more NG data "
              "or pick a lower threshold.")

    out_dir = pred.run_dir / "eval"
    out_dir.mkdir(exist_ok=True)
    ts = np.linspace(0, 1, 501)
    curves = {"val": (ts, *sweep(val_s, val_y, ts)), "test": (ts, *sweep(test_s, test_y, ts))}
    plot_threshold_curve(curves, target, chosen, out_dir / "threshold_curve.png")
    plot_confusion(results["test"]["at_chosen"]["confusion_matrix"], labels,
                   f"TEST @ threshold {chosen:.4f}", out_dir / "confusion_matrix.png")
    with open(out_dir / "eval.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out_dir / 'eval.json'}, threshold_curve.png, confusion_matrix.png")

    if args.save_threshold:
        path = pred.run_dir / "threshold.json"
        path.write_text(json.dumps({"threshold": chosen, "target_ng_recall": target,
                                    "picked_on": "val"}, indent=2), encoding="utf-8")
        print(f"saved threshold {chosen:.4f} -> {path}")


if __name__ == "__main__":
    main()
