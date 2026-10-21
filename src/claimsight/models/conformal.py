import math
from typing import Any

import numpy as np
import pandas as pd


def fit_conformal(
    model: Any,
    X_calib: pd.DataFrame | np.ndarray,
    y_calib: pd.Series | np.ndarray,
    alpha: float = 0.10,
) -> float:
    """Compute nonconformity calibration quantile on the log1p scale."""
    preds = np.asarray(model.predict(X_calib))
    y_true = np.asarray(y_calib)

    log_preds = np.log1p(np.clip(preds, 0, None))
    log_y = np.log1p(np.clip(y_true, 0, None))

    residuals = np.abs(log_y - log_preds)
    n = len(residuals)
    if n == 0:
        raise ValueError("Calibration set cannot be empty")

    rank = math.ceil((n + 1) * (1.0 - alpha))
    level = min(1.0, rank / n)

    # Compute empirical conformal quantile
    q = float(np.quantile(residuals, level, method="higher"))
    return q


def predict_interval(
    model: Any,
    X: pd.DataFrame | np.ndarray,
    q: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return lower and upper conformal prediction intervals in original dollar units."""
    preds = np.asarray(model.predict(X))
    log_preds = np.log1p(np.clip(preds, 0, None))

    lower_log = log_preds - q
    upper_log = log_preds + q

    lower = np.clip(np.expm1(lower_log), 0.0, None)
    upper = np.clip(np.expm1(upper_log), 0.0, None)

    return lower, upper


def evaluate_conformal_coverage(
    y_true: pd.Series | np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    """Calculate empirical coverage fraction on held-out evaluation rows."""
    y = np.asarray(y_true)
    covered = (y >= lower) & (y <= upper)
    return float(np.mean(covered))
