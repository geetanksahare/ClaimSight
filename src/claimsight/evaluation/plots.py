from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve


def plot_predicted_vs_actual(
    data: dict[str, np.ndarray],
    out_path: Path,
    title: str = "Predicted vs Actual Charges",
) -> Path:
    y_true = np.asarray(data["y_true"])
    y_pred = np.asarray(data["y_pred"])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(y_true, y_pred, alpha=0.5, edgecolors="none", c="#1f77b4")
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=1.5, label="Perfect Fit")

    ax.set_xlabel("Actual Charges ($)")
    ax.set_ylabel("Predicted Charges ($)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.6)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_residuals(
    data: dict[str, np.ndarray],
    out_path: Path,
) -> Path:
    y_true = np.asarray(data["y_true"])
    y_pred = np.asarray(data["y_pred"])
    residuals = y_pred - y_true

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.scatter(y_pred, residuals, alpha=0.5, c="#2ca02c", edgecolors="none")
    ax.axhline(0, color="red", linestyle="--", lw=1.5)
    ax.set_xlabel("Predicted Charges ($)")
    ax.set_ylabel("Residual (Predicted - Actual) ($)")
    ax.set_title("Residuals vs Predicted")
    ax.grid(True, linestyle=":", alpha=0.6)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_error_by_segment(
    data: dict[str, Any],
    out_path: Path,
    segment_name: str = "Segment",
) -> Path:
    # data format: {"categories": [...], "mae": [...], "rmse": [...]}
    categories = data["categories"]
    mae = data["mae"]
    rmse = data["rmse"]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(categories))
    width = 0.35

    ax.bar(x - width / 2, mae, width, label="MAE", color="#4e79a7")
    ax.bar(x + width / 2, rmse, width, label="RMSE", color="#e15759")

    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=30, ha="right")
    ax.set_ylabel("Error ($)")
    ax.set_title(f"Prediction Error by {segment_name}")
    ax.legend()
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_interval_coverage(
    data: dict[str, np.ndarray],
    out_path: Path,
) -> Path:
    y_true = np.asarray(data["y_true"])
    lower = np.asarray(data["lower"])
    upper = np.asarray(data["upper"])

    sorted_idx = np.argsort(y_true)
    y_s = y_true[sorted_idx]
    l_s = lower[sorted_idx]
    u_s = upper[sorted_idx]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))

    n_sample = min(100, len(y_s))
    step = len(y_s) // n_sample
    sub_idx = np.arange(0, len(y_s), step)[:n_sample]

    indices = np.arange(len(sub_idx))
    ax.fill_between(
        indices, l_s[sub_idx], u_s[sub_idx], color="skyblue", alpha=0.4, label="Prediction Interval"
    )
    ax.plot(indices, y_s[sub_idx], "k.", label="Actual Cost")

    covered = ((y_true >= lower) & (y_true <= upper)).mean() * 100
    ax.set_title(f"Conformal Prediction Intervals (Empirical Coverage: {covered:.1f}%)")
    ax.set_xlabel("Sample Index (sorted by cost)")
    ax.set_ylabel("Dollars ($)")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.6)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_pr_curve(
    data: dict[str, np.ndarray],
    out_path: Path,
) -> Path:
    y_true = np.asarray(data["y_true"])
    probs = np.asarray(data["probs"])

    precision, recall, _ = precision_recall_curve(y_true, probs)
    base_rate = float(np.mean(y_true))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))

    ax.plot(recall, precision, color="#1f77b4", lw=2, label="Precision-Recall Curve")
    ax.axhline(base_rate, color="navy", linestyle="--", label=f"Base Rate ({base_rate:.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.legend(loc="upper right")
    ax.grid(True, linestyle=":", alpha=0.6)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_roc_curve(
    data: dict[str, np.ndarray],
    out_path: Path,
) -> Path:
    y_true = np.asarray(data["y_true"])
    probs = np.asarray(data["probs"])

    fpr, tpr, _ = roc_curve(y_true, probs)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))

    ax.plot(fpr, tpr, color="#ff7f0e", lw=2, label="ROC Curve")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic (ROC)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.legend(loc="lower right")
    ax.grid(True, linestyle=":", alpha=0.6)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_calibration_curve(
    data: dict[str, np.ndarray],
    out_path: Path,
) -> Path:
    y_true = np.asarray(data["y_true"])
    probs = np.asarray(data["probs"])

    prob_true, prob_pred = calibration_curve(y_true, probs, n_bins=10, strategy="quantile")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))

    ax.plot(prob_pred, prob_true, "s-", color="#2ca02c", label="Calibration Curve")
    ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Curve (10 Quantile Bins)")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle=":", alpha=0.6)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
