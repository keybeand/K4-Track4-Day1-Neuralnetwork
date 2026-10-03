"""optimizer.py — Xây dựng bộ tối ưu và gradient clipping.
"""
from __future__ import annotations

import torch

OPTIMIZERS = ("sgd", "sgd_momentum", "adam", "adamw")


def build_optimizer(name: str, params, lr: float, weight_decay: float = 0.0,
                    momentum: float = 0.9, betas=(0.9, 0.999), eps: float = 1e-8):
    """Trả về một torch.optim.Optimizer."""
    if name not in OPTIMIZERS:
        raise ValueError(f"Optimizer {name} không hợp lệ. Chọn một trong {OPTIMIZERS}")

    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay)
    elif name == "sgd_momentum":
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif name == "adam":
        return torch.optim.Adam(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    elif name == "adamw":
        return torch.optim.AdamW(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)


def build_scheduler(optimizer, name: str | None, total_steps: int, **kwargs):
    """(Tuỳ chọn) Bộ lập lịch tốc độ học."""
    if name is None:
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    return None


def clip_gradients(params, max_norm: float | None) -> float:
    """Cắt gradient theo chuẩn L2 toàn cục, và TRẢ VỀ chuẩn gradient TRƯỚC KHI cắt."""
    if max_norm is None:
        total_norm = torch.nn.utils.clip_grad_norm_(params, float('inf'))
    else:
        total_norm = torch.nn.utils.clip_grad_norm_(params, max_norm)
    return float(total_norm)
