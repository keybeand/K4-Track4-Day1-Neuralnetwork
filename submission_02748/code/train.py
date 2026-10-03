"""train.py — Vòng huấn luyện, đánh giá và dự đoán mô hình.
"""
from __future__ import annotations

import random
import time
import numpy as np
import torch
import torch.nn.functional as F

from data import iterate_batches
from model import MLP, EXPECTED_PARAMS, count_params
from optimizer import build_optimizer, clip_gradients

DEFAULT_CFG = dict(
    exp_id="base-s1", group="baseline", description="Baseline M-base",
    loss="ce",                 # "ce" | "mse"
    optimizer="sgd_momentum",  # "sgd" | "sgd_momentum" | "adam" | "adamw"
    lr=0.05,
    weight_decay=0.0, momentum=0.9,
    batch=512, epochs=20,
    hidden=(256, 128), dropout=0.0, init="he",
    clip_norm=None,            # None = không clip
    precision="fp32",          # "fp32" | "fp16" | "bf16"
    seed=1,
)


def set_seed(seed: int) -> None:
    """Đặt seed cho random, numpy, torch (và torch.cuda nếu có)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def macro_f1_from_confusion(cm: np.ndarray) -> float:
    """macro-F1 = trung bình cộng F1 của 7 lớp; F1_c = 2PR/(P+R)."""
    f1_list = []
    for c in range(7):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        f1_list.append(f1)

    return float(np.mean(f1_list))


@torch.no_grad()
def predict(model, X, batch_size: int = 8192) -> torch.Tensor:
    """Trả về nhãn dự đoán int64 (N,) = argmax của logits."""
    model.eval()
    preds = []
    N = len(X)
    for i in range(0, N, batch_size):
        xb = X[i:i + batch_size]
        logits = model(xb)
        preds.append(torch.argmax(logits, dim=1))
    return torch.cat(preds, dim=0)


def compute_loss(logits, y, loss_name: str):
    """"ce"  : cross-entropy nhận logit thô và nhãn int64.
       "mse" : MSE giữa logit và one-hot của y.
    """
    if loss_name == "ce":
        return F.cross_entropy(logits, y)
    elif loss_name == "mse":
        y_onehot = F.one_hot(y, num_classes=7).float()
        return F.mse_loss(logits, y_onehot)
    else:
        raise ValueError(f"Unknown loss_name: {loss_name}")


@torch.no_grad()
def evaluate(model, X, y, loss_name: str = "ce", batch_size: int = 8192) -> dict:
    """Trả về dict(loss, acc, macro_f1) ở chế độ eval() (dropout tắt) và no_grad."""
    model.eval()
    N = len(X)
    total_loss = 0.0
    preds_list = []

    for i in range(0, N, batch_size):
        xb = X[i:i + batch_size]
        yb = y[i:i + batch_size]
        logits = model(xb)
        loss = compute_loss(logits, yb, loss_name)
        total_loss += loss.item() * len(xb)
        preds_list.append(torch.argmax(logits, dim=1))

    avg_loss = total_loss / N
    all_preds = torch.cat(preds_list, dim=0).cpu().numpy()
    y_true = y.cpu().numpy()

    acc = float((all_preds == y_true).mean())

    # Dựng ma trận nhầm lẫn
    cm = np.zeros((7, 7), dtype=np.int64)
    for t, p in zip(y_true, all_preds):
        cm[t, p] += 1

    macro_f1 = macro_f1_from_confusion(cm)

    return {"loss": avg_loss, "acc": acc, "macro_f1": macro_f1, "cm": cm}


def run_experiment(cfg: dict, data: dict) -> dict:
    """Huấn luyện một cấu hình và trả về lịch sử + tóm tắt."""
    set_seed(cfg["seed"])

    device = data["X_tr"].device
    hidden = tuple(cfg["hidden"])

    model = MLP(
        hidden=hidden,
        dropout=cfg["dropout"],
        init=cfg["init"],
        in_features=54,
        num_classes=7
    ).to(device)

    # Check số tham số
    expected = EXPECTED_PARAMS.get(hidden)
    if expected is not None:
        actual = count_params(model)
        assert actual == expected, f"Param count mismatch for {hidden}: got {actual}, expected {expected}"

    optimizer = build_optimizer(
        cfg["optimizer"],
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
        momentum=cfg["momentum"]
    )

    precision = cfg["precision"]
    scaler = torch.amp.GradScaler("cuda") if precision == "fp16" and device.type == "cuda" else None

    # Step 0 loss (val)
    step0_eval = evaluate(model, data["X_val"], data["y_val"], loss_name=cfg["loss"])
    step0_loss = step0_eval["loss"]

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "val_macro_f1": [],
        "grad_norm": [],
        "epoch_time_s": []
    }

    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    diverged = False

    gen = torch.Generator(device=device)
    gen.manual_seed(cfg["seed"])

    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()
        model.train()
        grad_norms = []

        for xb, yb in iterate_batches(data["X_tr"], data["y_tr"], cfg["batch"], generator=gen, shuffle=True):
            optimizer.zero_grad(set_to_none=True)

            if precision in ("fp16", "bf16") and device.type == "cuda":
                dtype = torch.float16 if precision == "fp16" else torch.bfloat16
                with torch.autocast("cuda", dtype=dtype):
                    logits = model(xb)
                    loss = compute_loss(logits, yb, cfg["loss"])
            else:
                logits = model(xb)
                loss = compute_loss(logits, yb, cfg["loss"])

            if torch.isnan(loss) or torch.isinf(loss):
                diverged = True
                break

            if scaler is not None:
                scaler.scale(loss).backward()
                if cfg["clip_norm"] is not None:
                    scaler.unscale_(optimizer)
                gn = clip_gradients(model.parameters(), cfg["clip_norm"])
                grad_norms.append(gn)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                gn = clip_gradients(model.parameters(), cfg["clip_norm"])
                grad_norms.append(gn)
                optimizer.step()

        if diverged:
            print(f"[{cfg['exp_id']}] Diverged at epoch {epoch}")
            break

        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()
        epoch_time = t1 - t0

        # Eval cuoi epoch (do train_loss trên 50 000 mau co dinh cua train de tiet kiem thoi gian)
        X_tr_sub = data["X_tr"][:50000]
        y_tr_sub = data["y_tr"][:50000]
        tr_eval = evaluate(model, X_tr_sub, y_tr_sub, loss_name=cfg["loss"])
        val_eval = evaluate(model, data["X_val"], data["y_val"], loss_name=cfg["loss"])

        avg_grad_norm = float(np.mean(grad_norms)) if grad_norms else 0.0

        history["epoch"].append(epoch)
        history["train_loss"].append(tr_eval["loss"])
        history["val_loss"].append(val_eval["loss"])
        history["val_acc"].append(val_eval["acc"])
        history["val_macro_f1"].append(val_eval["macro_f1"])
        history["grad_norm"].append(avg_grad_norm)
        history["epoch_time_s"].append(epoch_time)

        if val_eval["loss"] < best_val_loss:
            best_val_loss = val_eval["loss"]
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Summary
    if not diverged and best_epoch > 0:
        best_idx = history["epoch"].index(best_epoch)
        best_val_acc = history["val_acc"][best_idx]
        best_val_f1 = history["val_macro_f1"][best_idx]
        final_tr_loss = history["train_loss"][-1]
        final_val_loss = history["val_loss"][-1]
    else:
        best_val_acc = 0.0
        best_val_f1 = 0.0
        final_tr_loss = float("nan")
        final_val_loss = float("nan")

    avg_epoch_time = float(np.mean(history["epoch_time_s"])) if history["epoch_time_s"] else 0.0
    peak_mem = (torch.cuda.max_memory_allocated(device) / (1024 ** 2)) if device.type == "cuda" else 0.0

    summary = {
        "step0_loss": step0_loss,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "final_train_loss": final_tr_loss,
        "final_val_loss": final_val_loss,
        "val_acc": best_val_acc,
        "val_macro_f1": best_val_f1,
        "time_per_epoch_s": avg_epoch_time,
        "peak_mem_MB": peak_mem,
        "diverged": diverged
    }

    return {
        "cfg": cfg,
        "history": history,
        "summary": summary,
        "best_state": best_state
    }


def write_predictions(row_id, preds, path: str) -> None:
    """Ghi file nộp cho scripts/evaluate.py: CSV có tiêu đề `row_id,pred`."""
    import pandas as pd
    df = pd.DataFrame({"row_id": row_id, "pred": preds})
    df.to_csv(path, index=False)
    print(f"Predictions saved to {path} ({len(df)} rows)")


def final_eval(cfg: dict, result: dict, data: dict, pred_path: str) -> None:
    """Dùng MỘT LẦN cho cấu hình cuối cùng (và baseline): nạp best_state, dự đoán eval, ghi predictions."""
    device = data["X_tr"].device
    model = MLP(
        hidden=tuple(cfg["hidden"]),
        dropout=cfg["dropout"],
        init=cfg["init"],
        in_features=54,
        num_classes=7
    ).to(device)

    model.load_state_dict({k: v.to(device) for k, v in result["best_state"].items()})
    preds = predict(model, data["X_eval"])
    write_predictions(data["eval_row_id"], preds.cpu().numpy(), pred_path)
