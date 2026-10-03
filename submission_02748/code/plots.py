"""plots.py — Vẽ đồ thị thí nghiệm và đồ thị so sánh.
"""
from __future__ import annotations

import os
import matplotlib.pyplot as plt


def plot_run(result: dict, path: str) -> None:
    """Vẽ MỘT thí nghiệm thành một ảnh PNG có 3 ô:
         (1) train_loss và val_loss theo epoch
         (2) val_acc và val_macro_f1 theo epoch
         (3) grad_norm theo epoch (đo TRƯỚC khi clip)
    """
    history = result["history"]
    cfg = result["cfg"]
    summary = result["summary"]

    epochs = history["epoch"]
    if not epochs:
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # Ô 1: Loss
    axes[0].plot(epochs, history["train_loss"], label="Train Loss", color="blue", linestyle="--")
    axes[0].plot(epochs, history["val_loss"], label="Val Loss", color="red")
    if summary.get("best_epoch", 0) > 0:
        axes[0].axvline(summary["best_epoch"], color="gray", linestyle=":", label=f"Best Epoch ({summary['best_epoch']})")
    axes[0].set_title("Loss Trajectory")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Ô 2: Metrics (Acc & F1)
    axes[1].plot(epochs, history["val_acc"], label="Val Accuracy", color="green")
    axes[1].plot(epochs, history["val_macro_f1"], label="Val Macro-F1", color="purple")
    if summary.get("best_epoch", 0) > 0:
        axes[1].axvline(summary["best_epoch"], color="gray", linestyle=":", label=f"Best Epoch ({summary['best_epoch']})")
    axes[1].set_title("Validation Metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Ô 3: Gradient Norm
    axes[2].plot(epochs, history["grad_norm"], label="Grad Norm (pre-clip)", color="orange")
    axes[2].set_title("Gradient Norm (L2)")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Norm")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    # Title tổng quan
    exp_id = cfg["exp_id"]
    opt_info = f"opt={cfg['optimizer']}, lr={cfg['lr']}, batch={cfg['batch']}, init={cfg['init']}"
    fig.suptitle(f"Experiment: {exp_id} ({opt_info})", fontsize=12, fontweight="bold")

    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_compare(results: list[dict], metric: str, path: str, title: str = "") -> None:
    """Vẽ chồng một chỉ số (ví dụ "val_loss", "val_macro_f1", "grad_norm") của nhiều thí nghiệm."""
    if not results:
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    for res in results:
        cfg = res["cfg"]
        history = res["history"]
        if metric in history:
            ax.plot(history["epoch"], history[metric], label=cfg["exp_id"])

    ax.set_title(title if title else f"Comparison: {metric}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
