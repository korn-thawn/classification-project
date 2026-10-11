"""Train the OK/NG CNN from scratch.

Outputs runs/{model_version}/: best.pt (weights + config + split), config.yaml,
history.json, loss.png; and updates runs/LATEST to point at this run.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from vision.augment import Augmenter
from vision.config import load_config, resolve_device, set_seed
from vision.dataset import OkNgDataset, class_counts, collect, scan_sessions, split_by_session, summarize
from vision.model import build_model, count_params


def run_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device,
              optimizer: torch.optim.Optimizer | None = None) -> tuple[float, float]:
    """One pass over loader; trains if optimizer is given. Returns (mean loss, accuracy)."""
    training = optimizer is not None
    model.train(training)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            n += len(y)
    return total_loss / max(n, 1), correct / max(n, 1)


def plot_history(history: list[dict[str, float]], path: Path) -> None:
    """Save train/val loss and accuracy curves."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ep = [h["epoch"] for h in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(ep, [h["train_loss"] for h in history], label="train")
    ax1.plot(ep, [h["val_loss"] for h in history], label="val")
    ax1.set_title("loss (class-weighted CE)")
    ax1.set_xlabel("epoch")
    ax1.legend()
    ax2.plot(ep, [h["train_acc"] for h in history], label="train")
    ax2.plot(ep, [h["val_acc"] for h in history], label="val")
    ax2.set_title("accuracy")
    ax2.set_xlabel("epoch")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def seed_worker(worker_id: int) -> None:
    """Give each DataLoader worker its own augmentation RNG stream."""
    info = torch.utils.data.get_worker_info()
    if info is not None and info.dataset.augmenter is not None:
        info.dataset.augmenter.rng = np.random.default_rng(torch.initial_seed() % 2**32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--data-dir", help="override data.raw_dir")
    parser.add_argument("--epochs", type=int, help="override train.epochs")
    parser.add_argument("--device", help="override train.device (auto|cpu|cuda)")
    args = parser.parse_args()

    cfg: dict[str, Any] = load_config(args.config)
    if args.data_dir:
        cfg["data"]["raw_dir"] = args.data_dir
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    if args.device:
        cfg["train"]["device"] = args.device
    t_cfg = cfg["train"]
    classes = cfg["classes"]

    set_seed(cfg["seed"])
    device = resolve_device(t_cfg["device"])
    print(f"device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    sessions = scan_sessions(cfg["data"]["raw_dir"], classes)
    split = split_by_session(sessions, cfg["data"]["val_sessions"], cfg["data"]["test_sessions"])
    print(summarize(split, sessions, classes))
    train_items, val_items = collect(sessions, split["train"]), collect(sessions, split["val"])

    counts = class_counts(train_items, len(classes))
    if (counts == 0).any():
        raise SystemExit(f"train split is missing a class: {dict(zip(classes, counts.tolist()))}")
    class_weight = counts.sum() / (len(classes) * counts)
    print("class weights:", {c: round(float(w), 3) for c, w in zip(classes, class_weight)})

    cache = cfg["data"]["cache_in_ram"]
    train_ds = OkNgDataset(train_items, cfg, Augmenter(cfg["augment"], seed=cfg["seed"]), cache=cache)
    val_ds = OkNgDataset(val_items, cfg, augmenter=None, cache=cache)
    pin = device.type == "cuda"
    g = torch.Generator().manual_seed(cfg["seed"])
    train_dl = DataLoader(train_ds, batch_size=t_cfg["batch_size"], shuffle=True, generator=g,
                          num_workers=t_cfg["num_workers"], worker_init_fn=seed_worker, pin_memory=pin)
    val_dl = DataLoader(val_ds, batch_size=t_cfg["batch_size"], shuffle=False,
                        num_workers=t_cfg["num_workers"], pin_memory=pin)

    model = build_model(cfg).to(device)
    print(f"model: {count_params(model):,} params")
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weight, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=t_cfg["lr"], weight_decay=t_cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5,
                                                           patience=max(1, t_cfg["patience"] // 3))

    model_version = datetime.now().strftime("cnn-%Y%m%d-%H%M%S")
    out_dir = Path(t_cfg["out_dir"]) / model_version
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    history: list[dict[str, float]] = []
    best_loss, best_epoch, bad_epochs = float("inf"), 0, 0
    t_start = time.perf_counter()
    for epoch in range(1, t_cfg["epochs"] + 1):
        t0 = time.perf_counter()
        tr_loss, tr_acc = run_epoch(model, train_dl, criterion, device, optimizer)
        va_loss, va_acc = run_epoch(model, val_dl, criterion, device)
        scheduler.step(va_loss)
        lr = optimizer.param_groups[0]["lr"]
        history.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
                        "val_loss": va_loss, "val_acc": va_acc, "lr": lr})
        improved = va_loss < best_loss - 1e-4
        print(f"epoch {epoch:3d} | train {tr_loss:.4f} acc {tr_acc:.3f} | "
              f"val {va_loss:.4f} acc {va_acc:.3f} | lr {lr:.1e} | "
              f"{time.perf_counter() - t0:.1f}s{'  *' if improved else ''}")
        if improved:
            best_loss, best_epoch, bad_epochs = va_loss, epoch, 0
            torch.save({
                "model_state": model.state_dict(),
                "config": cfg,
                "classes": classes,
                "model_version": model_version,
                "epoch": epoch,
                "val_loss": va_loss,
                "split": split,
                "train_class_counts": counts.tolist(),
            }, out_dir / "best.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= t_cfg["patience"]:
                print(f"early stopping: val loss not improved for {bad_epochs} epochs")
                break

    with open(out_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    plot_history(history, out_dir / "loss.png")
    (Path(t_cfg["out_dir"]) / "LATEST").write_text(model_version, encoding="utf-8")
    print(f"done in {time.perf_counter() - t_start:.0f}s; best epoch {best_epoch} "
          f"val loss {best_loss:.4f} -> {out_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
