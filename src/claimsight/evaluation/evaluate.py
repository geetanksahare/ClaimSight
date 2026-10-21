import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
)


def _convert_numpy(obj: Any) -> Any:
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _convert_numpy(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_convert_numpy(v) for v in obj]
    return obj


def evaluate_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    segments: dict[str, pd.Series] | None = None,
) -> dict[str, Any]:
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)

    rmse = float(root_mean_squared_error(y_t, y_p))
    mae = float(mean_absolute_error(y_t, y_p))
    r2 = float(r2_score(y_t, y_p))
    med_ae = float(np.median(np.abs(y_t - y_p)))
    denom = np.clip(np.abs(y_t), 1.0, None)
    mape = float(np.mean(np.abs((y_t - y_p) / denom))) * 100.0

    metrics: dict[str, Any] = {
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "r2": round(r2, 4),
        "median_ae": round(med_ae, 4),
        "mape": round(mape, 4),
        "n_samples": int(len(y_t)),
    }

    if segments:
        seg_metrics: dict[str, dict[str, Any]] = {}
        for seg_name, seg_series in segments.items():
            seg_s = pd.Series(seg_series).reset_index(drop=True)
            seg_metrics[seg_name] = {}
            for val in seg_s.unique():
                mask = (seg_s == val).to_numpy()
                if mask.sum() == 0:
                    continue
                sub_t = y_t[mask]
                sub_p = y_p[mask]
                sub_denom = np.clip(np.abs(sub_t), 1.0, None)
                seg_metrics[seg_name][str(val)] = {
                    "count": int(mask.sum()),
                    "rmse": round(float(root_mean_squared_error(sub_t, sub_p)), 4),
                    "mae": round(float(mean_absolute_error(sub_t, sub_p)), 4),
                    "r2": round(float(r2_score(sub_t, sub_p)) if len(sub_t) > 1 else 0.0, 4),
                    "mape": round(float(np.mean(np.abs((sub_t - sub_p) / sub_denom))) * 100.0, 4),
                }
        metrics["segments"] = seg_metrics

    return metrics


def evaluate_classification(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float | None = None,
) -> dict[str, Any]:
    y_t = np.asarray(y_true, dtype=int)
    probs = np.asarray(y_prob, dtype=float)

    threshold_used = 0.5 if threshold is None else float(threshold)
    threshold_note = "default (0.5)" if threshold is None else "custom"
    y_pred = (probs >= threshold_used).astype(int)

    roc_auc = float(roc_auc_score(y_t, probs))
    pr_auc = float(average_precision_score(y_t, probs))
    f1 = float(f1_score(y_t, y_pred, zero_division=0))
    prec = float(precision_score(y_t, y_pred, zero_division=0))
    rec = float(recall_score(y_t, y_pred, zero_division=0))
    brier = float(brier_score_loss(y_t, probs))
    cm = confusion_matrix(y_t, y_pred).tolist()
    base_rate = float(np.mean(y_t))

    # Top 10% ranked rows
    n = len(y_t)
    k = max(1, int(0.10 * n))
    top_indices = np.argsort(probs)[::-1][:k]
    prec_at_k = float(np.mean(y_t[top_indices]))
    total_positives = int(np.sum(y_t))
    rec_at_k = float(np.sum(y_t[top_indices]) / max(total_positives, 1))

    return {
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "f1": round(f1, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "brier": round(brier, 4),
        "threshold_used": round(threshold_used, 4),
        "threshold_note": threshold_note,
        "base_rate": round(base_rate, 4),
        "precision_at_top_k": round(prec_at_k, 4),
        "recall_at_top_k": round(rec_at_k, 4),
        "confusion_matrix": cm,
        "n_samples": int(n),
    }


def evaluate(
    y_true: Any,
    y_pred: Any,
    task: Literal["regression", "classification"],
    y_prob: Any = None,
    threshold: float | None = None,
    segments: dict[str, pd.Series] | None = None,
) -> dict[str, Any]:
    """Calculate performance metrics on held-out evaluation sets."""
    if task == "regression":
        results = evaluate_regression(y_true, y_pred, segments=segments)
    elif task == "classification":
        if y_prob is None:
            y_prob = y_pred
        results = evaluate_classification(y_true, y_prob, threshold=threshold)
    else:
        raise ValueError(f"Unsupported task type: {task}")

    return _convert_numpy(results)


def save_metrics(metrics: dict[str, Any], path: Path) -> None:
    """Serialize evaluated metrics to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
