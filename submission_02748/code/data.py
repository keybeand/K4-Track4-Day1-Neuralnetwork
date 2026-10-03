"""data.py — Chuẩn bị dữ liệu cho bài lab Neural Network Day 1.

Nhiệm vụ: nạp tập train/eval đã chia sẵn, tách validation từ train, chuẩn hoá, đưa lên thiết bị.
"""
from __future__ import annotations

import os
import numpy as np
import torch
from sklearn.model_selection import train_test_split

N_NUMERIC = 10  # số cột liên tục cần chuẩn hoá (cột 0..9)


def load_split(processed_dir: str = "data/processed"):
    """Nạp train và eval từ file .npz.

    Trả về: X_train_full, y_train_full, X_eval, y_eval, eval_row_id
    """
    train_path = os.path.join(processed_dir, "train.npz")
    eval_path = os.path.join(processed_dir, "eval.npz")

    train_data = np.load(train_path)
    eval_data = np.load(eval_path)

    X_train_full = train_data["X"].astype(np.float32)
    y_train_full = train_data["y"].astype(np.int64)

    X_eval = eval_data["X"].astype(np.float32)
    y_eval = eval_data["y"].astype(np.int64)
    eval_row_id = eval_data["row_id"]

    assert X_train_full.shape == (464809, 54), f"X_train_full shape sai: {X_train_full.shape}"
    assert y_train_full.shape == (464809,), f"y_train_full shape sai: {y_train_full.shape}"
    assert X_eval.shape == (116203, 54), f"X_eval shape sai: {X_eval.shape}"
    assert y_eval.shape == (116203,), f"y_eval shape sai: {y_eval.shape}"

    return X_train_full, y_train_full, X_eval, y_eval, eval_row_id


def make_val_split(X, y, val_fraction: float = 0.2, seed: int = 42):
    """Tách validation TỪ train (không đụng eval). Phân tầng theo nhãn."""
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=val_fraction, stratify=y, random_state=seed
    )
    return X_tr, y_tr, X_val, y_val


def fit_standardizer(X_tr):
    """Tính mean và std của N_NUMERIC cột đầu CHỈ trên tập train (sau khi tách val)."""
    X_num = X_tr[:, :N_NUMERIC]
    mean = X_num.mean(axis=0)
    std = X_num.std(axis=0)
    # Tránh chia cho 0 nếu std nhỏ
    std[std == 0] = 1.0
    return mean, std


def apply_standardizer(X, mean, std):
    """Trả về bản sao của X, trong đó 10 cột đầu được (x - mean) / std; 44 cột nhị phân giữ nguyên."""
    X_scaled = X.copy()
    X_scaled[:, :N_NUMERIC] = (X_scaled[:, :N_NUMERIC] - mean) / std
    return X_scaled


def prepare_data(device: str, val_fraction: float = 0.2, seed: int = 42,
                 processed_dir: str = "data/processed") -> dict:
    """Gộp các bước trên và đưa TOÀN BỘ dữ liệu lên `device` một lần (không dùng DataLoader)."""
    X_train_full, y_train_full, X_eval, y_eval, eval_row_id = load_split(processed_dir)

    X_tr, y_tr, X_val, y_val = make_val_split(X_train_full, y_train_full, val_fraction=val_fraction, seed=seed)

    mean, std = fit_standardizer(X_tr)

    X_tr_scaled = apply_standardizer(X_tr, mean, std)
    X_val_scaled = apply_standardizer(X_val, mean, std)
    X_eval_scaled = apply_standardizer(X_eval, mean, std)

    # In thông tin các tập
    print(f"Data Prepared successfully:")
    print(f"  Train set: {X_tr_scaled.shape[0]} samples")
    print(f"  Val set  : {X_val_scaled.shape[0]} samples")
    print(f"  Eval set : {X_eval_scaled.shape[0]} samples")

    # Accuracy đoán đa số (lớp 1) trên val
    majority_class_acc = (y_val == 1).mean()
    print(f"  Majority class (class 1) accuracy on Val: {majority_class_acc:.4f}")

    dev = torch.device(device)
    data_dict = {
        "X_tr": torch.tensor(X_tr_scaled, dtype=torch.float32, device=dev),
        "y_tr": torch.tensor(y_tr, dtype=torch.int64, device=dev),
        "X_val": torch.tensor(X_val_scaled, dtype=torch.float32, device=dev),
        "y_val": torch.tensor(y_val, dtype=torch.int64, device=dev),
        "X_eval": torch.tensor(X_eval_scaled, dtype=torch.float32, device=dev),
        "y_eval": torch.tensor(y_eval, dtype=torch.int64, device=dev),
        "eval_row_id": eval_row_id,
        "mean": mean,
        "std": std
    }
    return data_dict


def iterate_batches(X, y, batch_size: int, generator: torch.Generator | None = None, shuffle: bool = True):
    """Generator trả về từng cặp (xb, yb), thay cho DataLoader."""
    N = len(X)
    if shuffle:
        perm = torch.randperm(N, generator=generator, device=X.device)
    else:
        perm = torch.arange(N, device=X.device)

    for i in range(0, N, batch_size):
        idx = perm[i:i + batch_size]
        yield X[idx], y[idx]
