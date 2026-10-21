import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    make_scorer,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.model_selection import cross_validate

from claimsight.config import get_settings
from claimsight.logging_config import get_logger

logger = get_logger(__name__)


def get_default_regression_scorers() -> dict[str, Any]:
    return {
        "rmse": make_scorer(root_mean_squared_error, greater_is_better=False),
        "mae": make_scorer(mean_absolute_error, greater_is_better=False),
        "r2": make_scorer(r2_score, greater_is_better=True),
    }


def get_default_classification_scorers() -> dict[str, Any]:
    return {
        "roc_auc": make_scorer(roc_auc_score, response_method="predict_proba"),
        "average_precision": make_scorer(average_precision_score, response_method="predict_proba"),
        "f1": make_scorer(f1_score, response_method="predict"),
        "brier": make_scorer(
            brier_score_loss, response_method="predict_proba", greater_is_better=False
        ),
    }


def train_and_compare(
    X: pd.DataFrame,
    y: pd.Series,
    zoo: dict[str, Any],
    cv: Any,
    scoring: dict[str, str | Callable],
    groups: pd.Series | None = None,
    experiment: str = "claimsight",
    primary_metric: str = "rmse",
) -> pd.DataFrame:
    """Evaluate candidate estimators across cross-validation splits and log to MLflow."""
    cfg = get_settings()
    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    mlflow.set_experiment(experiment)

    leaderboard_rows = []

    with mlflow.start_run(run_name=f"{experiment}_comparison") as parent_run:
        logger.info("Started parent MLflow run: %s (id: %s)", experiment, parent_run.info.run_id)

        for model_name, estimator in zoo.items():
            logger.info("Cross-validating %s...", model_name)
            start_time = time.time()

            # Run CV
            cv_results = cross_validate(
                estimator,
                X,
                y,
                cv=cv,
                scoring=scoring,
                groups=groups,
                return_train_score=True,
                n_jobs=-1,
                error_score="raise",
            )
            _ = time.time() - start_time
            mean_fit_time = float(np.mean(cv_results["fit_time"]))

            row_data: dict[str, Any] = {
                "model": model_name,
                "fit_time_mean": round(mean_fit_time, 4),
            }

            metrics_to_log: dict[str, float] = {"mean_fit_time": mean_fit_time}

            for metric_name in scoring.keys():
                test_scores = cv_results[f"test_{metric_name}"]
                train_scores = cv_results[f"train_{metric_name}"]

                # For error metrics where greater_is_better is False, flip sign for display
                if metric_name in ("rmse", "mae", "brier"):
                    test_scores = np.abs(test_scores)
                    train_scores = np.abs(train_scores)

                test_mean = float(np.mean(test_scores))
                test_std = float(np.std(test_scores))
                train_mean = float(np.mean(train_scores))

                row_data[f"{metric_name}_mean"] = round(test_mean, 4)
                row_data[f"{metric_name}_std"] = round(test_std, 4)
                row_data[f"{metric_name}_train"] = round(train_mean, 4)

                metrics_to_log[f"cv_{metric_name}_mean"] = test_mean
                metrics_to_log[f"cv_{metric_name}_std"] = test_std
                metrics_to_log[f"train_{metric_name}_mean"] = train_mean

            # Calculate overfit gap
            if "r2" in scoring:
                overfit_gap = row_data.get("r2_train", 0.0) - row_data.get("r2_mean", 0.0)
            elif "rmse" in scoring:
                test_rmse = row_data.get("rmse_mean", 1.0)
                train_rmse = row_data.get("rmse_train", 1.0)
                overfit_gap = (test_rmse - train_rmse) / max(test_rmse, 1e-6)
            elif "average_precision" in scoring:
                overfit_gap = row_data.get("average_precision_train", 0.0) - row_data.get(
                    "average_precision_mean", 0.0
                )
            else:
                overfit_gap = 0.0

            row_data["overfit_gap"] = round(overfit_gap, 4)
            row_data["overfit_flag"] = bool(overfit_gap > 0.20)
            metrics_to_log["overfit_gap"] = overfit_gap

            leaderboard_rows.append(row_data)

            # Log to child MLflow run
            with mlflow.start_run(run_name=model_name, nested=True):
                mlflow.log_params(
                    {"model_name": model_name, "cv_folds": getattr(cv, "n_splits", 5)}
                )
                mlflow.log_metrics(metrics_to_log)

        leaderboard_df = pd.DataFrame(leaderboard_rows)

        # Sort leaderboard best first
        metric_col = f"{primary_metric}_mean"
        if metric_col in leaderboard_df.columns:
            ascending = primary_metric in ("rmse", "mae", "brier")
            leaderboard_df = leaderboard_df.sort_values(
                by=metric_col, ascending=ascending
            ).reset_index(drop=True)

        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / f"{experiment}_leaderboard.csv"
            leaderboard_df.to_csv(csv_path, index=False)
            mlflow.log_artifact(str(csv_path))

    return leaderboard_df
