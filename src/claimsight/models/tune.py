import tempfile
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import optuna
import pandas as pd
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.base import clone
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
    root_mean_squared_error,
)

from claimsight.config import get_settings
from claimsight.logging_config import get_logger

logger = get_logger(__name__)

# Suppress Optuna logging verbosity in stdout
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _find_param_prefix(estimator: Any) -> str:
    params = estimator.get_params()
    if any(k.startswith("regressor__model__") for k in params):
        return "regressor__model__"
    if any(k.startswith("model__") for k in params):
        return "model__"
    return ""


def tune_model(
    name: str,
    base_estimator: Any,
    X: pd.DataFrame,
    y: pd.Series,
    cv: Any,
    scoring_metric: str,
    n_trials: int,
    seed: int,
    groups: pd.Series | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Tune estimator hyperparameters via Optuna Bayesian optimization."""
    cfg = get_settings()
    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)

    prefix = _find_param_prefix(base_estimator)
    is_minimizing = scoring_metric.lower() in ("rmse", "mae", "brier")
    direction = "minimize" if is_minimizing else "maximize"

    valid_param_keys = set(base_estimator.get_params().keys())

    def objective(trial: optuna.Trial) -> float:
        suggested: dict[str, Any] = {}

        if f"{prefix}n_estimators" in valid_param_keys:
            suggested["n_estimators"] = trial.suggest_int("n_estimators", 200, 1200, step=50)

        if f"{prefix}learning_rate" in valid_param_keys:
            suggested["learning_rate"] = trial.suggest_float("learning_rate", 0.01, 0.2, log=True)

        if f"{prefix}subsample" in valid_param_keys:
            suggested["subsample"] = trial.suggest_float("subsample", 0.6, 1.0, step=0.05)

        if f"{prefix}colsample_bytree" in valid_param_keys:
            suggested["colsample_bytree"] = trial.suggest_float(
                "colsample_bytree", 0.6, 1.0, step=0.05
            )

        if f"{prefix}reg_alpha" in valid_param_keys:
            suggested["reg_alpha"] = trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True)

        if f"{prefix}reg_lambda" in valid_param_keys:
            suggested["reg_lambda"] = trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True)

        # Architecture-specific
        if f"{prefix}num_leaves" in valid_param_keys:
            suggested["num_leaves"] = trial.suggest_int("num_leaves", 15, 63)
        if f"{prefix}min_child_samples" in valid_param_keys:
            suggested["min_child_samples"] = trial.suggest_int("min_child_samples", 5, 50)
        if f"{prefix}max_depth" in valid_param_keys:
            suggested["max_depth"] = trial.suggest_int("max_depth", 2, 7)
        if f"{prefix}min_child_weight" in valid_param_keys:
            suggested["min_child_weight"] = trial.suggest_int("min_child_weight", 1, 10)
        if f"{prefix}min_samples_leaf" in valid_param_keys:
            suggested["min_samples_leaf"] = trial.suggest_int("min_samples_leaf", 1, 10)

        model_params = {
            f"{prefix}{k}": v for k, v in suggested.items() if f"{prefix}{k}" in valid_param_keys
        }

        est = clone(base_estimator)
        est.set_params(**model_params)

        scores = []
        splits = cv.split(X, y, groups=groups)
        for fold, (train_idx, val_idx) in enumerate(splits):
            X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

            est.fit(X_tr, y_tr)

            if scoring_metric.lower() == "rmse":
                preds = est.predict(X_val)
                score = root_mean_squared_error(y_val, preds)
            elif scoring_metric.lower() in ("average_precision", "pr_auc"):
                probs = est.predict_proba(X_val)[:, 1]
                score = average_precision_score(y_val, probs)
            elif scoring_metric.lower() == "roc_auc":
                probs = est.predict_proba(X_val)[:, 1]
                score = roc_auc_score(y_val, probs)
            elif scoring_metric.lower() == "f1":
                preds = est.predict(X_val)
                score = f1_score(y_val, preds)
            else:
                preds = est.predict(X_val)
                score = root_mean_squared_error(y_val, preds)

            scores.append(score)
            trial.report(float(np.mean(scores)), step=fold)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(scores))

    sampler = TPESampler(seed=seed)
    pruner = MedianPruner()
    study = optuna.create_study(direction=direction, sampler=sampler, pruner=pruner)

    logger.info(
        "Starting Optuna study for %s (%d trials, metric: %s)...", name, n_trials, scoring_metric
    )
    study.optimize(objective, n_trials=n_trials, n_jobs=1)

    best_params = study.best_params
    best_value = study.best_value
    logger.info(
        "Best trial for %s: %s = %.4f with params %s", name, scoring_metric, best_value, best_params
    )

    # Recreate unfitted estimator with best params
    reconstructed_params = {
        f"{prefix}{k}": v for k, v in best_params.items() if f"{prefix}{k}" in valid_param_keys
    }
    best_estimator = clone(base_estimator)
    best_estimator.set_params(**reconstructed_params)

    # Log study history and best params to MLflow
    try:
        with mlflow.start_run(run_name=f"{name}_tuning"):
            mlflow.log_params(best_params)
            mlflow.log_metric(f"best_{scoring_metric}", best_value)
            trials_df = study.trials_dataframe()
            with tempfile.TemporaryDirectory() as tmp_dir:
                hist_path = Path(tmp_dir) / f"{name}_optuna_history.csv"
                trials_df.to_csv(hist_path, index=False)
                mlflow.log_artifact(str(hist_path))
    except Exception as e:
        logger.warning("MLflow logging skipped for tuning: %s", e)

    return best_estimator, best_params
