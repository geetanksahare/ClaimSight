from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split

from claimsight.config import get_settings
from claimsight.data.load_and_clean import load_and_clean
from claimsight.evaluation.evaluate import evaluate, save_metrics
from claimsight.evaluation.fairness_audit import fairness_audit
from claimsight.evaluation.plots import (
    plot_error_by_segment,
    plot_interval_coverage,
    plot_predicted_vs_actual,
    plot_residuals,
)
from claimsight.explain.explain_prediction import explain_global
from claimsight.features.build_features import build_features
from claimsight.features.cost_features import split_xy
from claimsight.logging_config import get_logger
from claimsight.models.conformal import evaluate_conformal_coverage, fit_conformal, predict_interval
from claimsight.models.registry import save_model
from claimsight.models.train_and_compare import get_default_regression_scorers, train_and_compare
from claimsight.models.tune import tune_model
from claimsight.models.zoo import get_model_zoo
from claimsight.reporting.model_card import generate_model_card

logger = get_logger(__name__)


def train_cost_pipeline(
    data_path: Path | None = None,
    optuna_trials: int | None = None,
) -> dict[str, Any]:
    """Execute end-to-end training, hyperparameter tuning, conformal calibration, and auditing for cost model."""
    cfg = get_settings()
    seed = cfg.seed

    # 1. Load and clean cost data
    if data_path is not None and data_path.exists():
        raw_df = (
            pd.read_parquet(data_path) if data_path.suffix == ".parquet" else pd.read_csv(data_path)
        )
    else:
        raw_df = load_and_clean("cost", save=True)

    # 2. Build feature table
    df_feat = build_features("cost", raw_df, cfg)

    # 3. Stratified split 70 / 15 / 15 on smoker
    test_size = cfg.cost.test_size
    calib_size = cfg.cost.calib_size

    # Split train+calib (85%) and test (15%)
    df_train_calib, df_test = train_test_split(
        df_feat,
        test_size=test_size,
        stratify=df_feat["smoker"],
        random_state=seed,
    )

    # Split train (70% overall) and calib (15% overall)
    calib_relative_ratio = calib_size / (1.0 - test_size)
    df_train, df_calib = train_test_split(
        df_train_calib,
        test_size=calib_relative_ratio,
        stratify=df_train_calib["smoker"],
        random_state=seed,
    )

    logger.info(
        "Cost splits created - Train: %d, Calib: %d, Test: %d",
        len(df_train),
        len(df_calib),
        len(df_test),
    )

    X_train, y_train = split_xy(df_train)
    X_calib, y_calib = split_xy(df_calib)
    X_test, y_test = split_xy(df_test)

    # 4. Model zoo and 5-fold cross validation comparison
    zoo = get_model_zoo("cost", seed=seed)
    cv = KFold(n_splits=cfg.cost.cv_folds, shuffle=True, random_state=seed)
    scorers = get_default_regression_scorers()

    leaderboard = train_and_compare(
        X_train,
        y_train,
        zoo=zoo,
        cv=cv,
        scoring=scorers,
        experiment="cost_regression",
        primary_metric="rmse",
    )
    logger.info(
        "Cost Model Leaderboard:\n%s", leaderboard[["model", "rmse_mean", "r2_mean", "overfit_gap"]]
    )

    # 5. Tune top 2 gradient boosting models with Optuna
    eligible_models = [m for m in ["lightgbm", "xgboost"] if m in leaderboard["model"].tolist()]
    trials_count = optuna_trials if optuna_trials is not None else cfg.cost.optuna_trials

    tuned_candidates: list[tuple[str, Any, float]] = []
    for model_name in eligible_models:
        base_est = zoo[model_name]
        tuned_est, best_params = tune_model(
            name=f"cost_{model_name}",
            base_estimator=base_est,
            X=X_train,
            y=y_train,
            cv=cv,
            scoring_metric="rmse",
            n_trials=trials_count,
            seed=seed,
        )
        # Compute tuned CV RMSE
        cv_scores = cross_val_score(
            tuned_est,
            X_train,
            y_train,
            cv=cv,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
        )
        tuned_rmse = float(np.mean(np.abs(cv_scores)))
        tuned_candidates.append((model_name, tuned_est, tuned_rmse))
        logger.info("Tuned %s CV RMSE: %.4f", model_name, tuned_rmse)

    # Select final model by lowest tuned CV RMSE
    tuned_candidates.sort(key=lambda item: item[2])
    best_model_name, final_estimator, best_tuned_rmse = tuned_candidates[0]
    logger.info("Selected final tuned model: %s (CV RMSE: %.4f)", best_model_name, best_tuned_rmse)

    # 6. Refit on train
    final_estimator.fit(X_train, y_train)

    # 7. Fit conformal prediction interval on calib
    alpha = cfg.cost.interval_alpha
    conformal_q = fit_conformal(final_estimator, X_calib, y_calib, alpha=alpha)
    logger.info(
        "Computed conformal nonconformity quantile q: %.4f (alpha: %.2f)", conformal_q, alpha
    )

    # 8. Single final evaluation on held-out test set
    y_test_pred = final_estimator.predict(X_test)
    lower_test, upper_test = predict_interval(final_estimator, X_test, conformal_q)

    test_metrics = evaluate(
        y_true=y_test,
        y_pred=y_test_pred,
        task="regression",
        segments={
            "smoker": X_test["smoker"],
            "sex": X_test["sex"],
            "region": X_test["region"],
        },
    )
    emp_coverage = evaluate_conformal_coverage(y_test, lower_test, upper_test)
    test_metrics["conformal_q"] = round(conformal_q, 4)
    test_metrics["conformal_coverage"] = round(emp_coverage, 4)
    test_metrics["target_coverage"] = round(1.0 - alpha, 4)
    test_metrics["best_model"] = best_model_name

    logger.info(
        "Final Test Evaluation: R2 = %.4f, RMSE = $%.2f, MAE = $%.2f, Coverage = %.1f%%",
        test_metrics["r2"],
        test_metrics["rmse"],
        test_metrics["mae"],
        emp_coverage * 100,
    )

    # Save metrics JSON
    metrics_path = cfg.paths.reports_dir / "metrics" / "cost.json"
    save_metrics(test_metrics, metrics_path)

    # 9. Generate figures
    fig_dir = cfg.paths.reports_dir / "figures"
    plot_predicted_vs_actual(
        {"y_true": y_test, "y_pred": y_test_pred},
        fig_dir / "cost_predicted_vs_actual.png",
    )
    plot_residuals(
        {"y_true": y_test, "y_pred": y_test_pred},
        fig_dir / "cost_residuals.png",
    )
    plot_interval_coverage(
        {"y_true": y_test, "lower": lower_test, "upper": upper_test},
        fig_dir / "cost_interval_coverage.png",
    )
    smoker_seg_data = {
        "categories": list(test_metrics["segments"]["smoker"].keys()),
        "mae": [
            test_metrics["segments"]["smoker"][k]["mae"] for k in test_metrics["segments"]["smoker"]
        ],
        "rmse": [
            test_metrics["segments"]["smoker"][k]["rmse"]
            for k in test_metrics["segments"]["smoker"]
        ],
    }
    plot_error_by_segment(
        smoker_seg_data, fig_dir / "cost_error_by_smoker.png", segment_name="Smoking Status"
    )

    # 10. Fairness Audit & Unaware Variant Comparison
    sensitive_cols = cfg.fairness.sensitive_cost
    fairness_results = fairness_audit(
        y_true=y_test,
        y_pred=y_test_pred,
        sensitive=X_test[sensitive_cols],
        task="regression",
        relative_gap_threshold=cfg.fairness.relative_gap_threshold,
        module="cost",
    )

    # Demographic-unaware model comparison (drop sex and region)
    unaware_features = [c for c in X_train.columns if c not in ("sex", "sex_flag", "region")]
    unaware_cat = [
        c
        for c in ["smoker", "age_band", "bmi_category", "family_size_bucket"]
        if c in unaware_features
    ]
    unaware_num = [c for c in X_train.columns if c in unaware_features and c not in unaware_cat]

    from sklearn.compose import TransformedTargetRegressor
    from sklearn.pipeline import Pipeline

    from claimsight.models.zoo import _create_preprocessor

    unaware_pipe = Pipeline(
        [
            ("preprocessor", _create_preprocessor(unaware_cat, unaware_num, scale_numeric=False)),
            ("model", clone(final_estimator.regressor_.named_steps["model"])),
        ]
    )
    unaware_model = TransformedTargetRegressor(
        regressor=unaware_pipe, func=np.log1p, inverse_func=np.expm1
    )
    unaware_model.fit(X_train[unaware_features], y_train)
    unaware_preds = unaware_model.predict(X_test[unaware_features])
    unaware_metrics = evaluate(y_test, unaware_preds, task="regression")
    logger.info(
        "Demographic-unaware model test RMSE: $%.2f (Baseline: $%.2f)",
        unaware_metrics["rmse"],
        test_metrics["rmse"],
    )

    # 11. Save model and metadata
    now_utc = datetime.now(timezone.utc)
    version = now_utc.strftime("%Y%m%d_%H%M%S")
    metadata = {
        "module": "cost",
        "version": version,
        "model_type": type(final_estimator.regressor_.named_steps["model"]).__name__,
        "features": list(X_train.columns),
        "training_rows": int(len(X_train)),
        "seed": seed,
        "conformal_q": float(conformal_q),
        "alpha": float(alpha),
        "metrics": test_metrics,
        "numeric_ranges": {
            col: {"min": float(X_train[col].min()), "max": float(X_train[col].max())}
            for col in ["age", "bmi", "children"]
        },
        "split_strategy": "Stratified on smoker, 70/15/15 (train/calib/test)",
        "unaware_variant_rmse": unaware_metrics["rmse"],
        "unaware_variant_mae": unaware_metrics["mae"],
    }
    save_model("cost", final_estimator, metadata)

    # 12. Generate Model Card and Global Explainability
    generate_model_card("cost")
    explain_global("cost")

    return {
        "model": final_estimator,
        "metadata": metadata,
        "test_metrics": test_metrics,
        "fairness_results": fairness_results,
    }
