from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from claimsight.config import get_settings
from claimsight.data.load_and_clean import load_and_clean
from claimsight.evaluation.evaluate import evaluate, save_metrics
from claimsight.evaluation.fairness_audit import fairness_audit
from claimsight.evaluation.plots import plot_calibration_curve, plot_pr_curve
from claimsight.features.build_features import build_features
from claimsight.features.member_features import label_high_cost, split_member_xy
from claimsight.logging_config import get_logger
from claimsight.models.registry import save_model
from claimsight.models.train_and_compare import (
    get_default_classification_scorers,
    get_default_regression_scorers,
    train_and_compare,
)
from claimsight.models.tune import tune_model
from claimsight.models.zoo import get_model_zoo
from claimsight.reporting.model_card import generate_model_card

logger = get_logger(__name__)


def train_member_pipeline(
    trials: int | None = None,
) -> dict[str, Any]:
    """Execute end-to-end longitudinal member risk modeling, calibration, and tier cutoff generation."""
    cfg = get_settings()
    seed = cfg.seed

    # 1. Load SynPUF tables
    processed_dir = cfg.paths.processed_dir
    bene_file = processed_dir / "synpuf_beneficiary.parquet"
    if not bene_file.exists():
        synpuf_data = load_and_clean("synpuf", save=True)
    else:
        synpuf_data = {
            "beneficiary": pd.read_parquet(bene_file),
            "inpatient": pd.read_parquet(processed_dir / "synpuf_inpatient.parquet"),
            "outpatient": pd.read_parquet(processed_dir / "synpuf_outpatient.parquet"),
            "carrier_agg": pd.read_parquet(processed_dir / "synpuf_carrier_agg.parquet"),
        }

    # 2. Build multi-year member feature dataset
    df_feat = build_features("member", synpuf_data, cfg)

    # 3. Grouped split by member_id: 80% train, 20% test (strict zero-leakage guarantee)
    gss = GroupShuffleSplit(n_splits=1, train_size=0.80, random_state=seed)
    train_idx, test_idx = next(gss.split(df_feat, groups=df_feat["member_id"]))

    df_train = df_feat.iloc[train_idx].copy().reset_index(drop=True)
    df_test = df_feat.iloc[test_idx].copy().reset_index(drop=True)

    # Assert no member overlap between train and test
    train_members = set(df_train["member_id"])
    test_members = set(df_test["member_id"])
    overlap = train_members.intersection(test_members)
    if overlap:
        raise ValueError(
            f"Data leakage detected! {len(overlap)} members found in both train and test splits"
        )

    logger.info(
        "Member split verified (0 member overlap) - Train: %d rows (%d members), Test: %d rows (%d members)",
        len(df_train),
        len(train_members),
        len(df_test),
        len(test_members),
    )

    X_train, y_train_cost = split_member_xy(df_train)
    X_test, y_test_cost = split_member_xy(df_test)

    # 4. Compute high-cost binary threshold on TRAIN only
    high_cost_q = cfg.member.high_cost_quantile
    high_cost_threshold, y_train_high_cost = label_high_cost(
        y_train_cost, y_train_cost, q=high_cost_q
    )
    _, y_test_high_cost = label_high_cost(y_train_cost, y_test_cost, q=high_cost_q)
    logger.info("High-cost cutoff (90th percentile of train): $%.2f", high_cost_threshold)

    # Determine column types for preprocessing
    cat_cols = [c for c in ["sex", "race"] if c in X_train.columns]
    num_cols = [c for c in X_train.columns if c not in cat_cols]

    # 5. Regression Leaderboard & Tuning
    cv_reg = GroupKFold(n_splits=cfg.member.cv_folds)
    reg_zoo = get_model_zoo("member_reg", seed=seed, cat_cols=cat_cols, num_cols=num_cols)
    reg_scorers = get_default_regression_scorers()

    reg_leaderboard = train_and_compare(
        X_train,
        y_train_cost,
        zoo=reg_zoo,
        cv=cv_reg,
        scoring=reg_scorers,
        groups=df_train["member_id"],
        experiment="member_cost_regression",
        primary_metric="rmse",
    )
    logger.info(
        "Member Cost Regression Leaderboard:\n%s",
        reg_leaderboard[["model", "rmse_mean", "r2_mean"]],
    )

    # Tune top regressor
    n_trials = trials if trials is not None else cfg.member.optuna_trials
    best_reg_name = [m for m in reg_leaderboard["model"].tolist() if m != "dummy"][0]
    tuned_reg, _ = tune_model(
        name=f"member_reg_{best_reg_name}",
        base_estimator=reg_zoo[best_reg_name],
        X=X_train,
        y=y_train_cost,
        cv=cv_reg,
        scoring_metric="rmse",
        n_trials=n_trials,
        seed=seed,
        groups=df_train["member_id"],
    )
    tuned_reg.fit(X_train, y_train_cost)

    # 6. Classification Leaderboard & Tuning
    cv_clf = GroupKFold(n_splits=cfg.member.cv_folds)
    clf_zoo = get_model_zoo("member_clf", seed=seed, cat_cols=cat_cols, num_cols=num_cols)
    clf_scorers = get_default_classification_scorers()

    clf_leaderboard = train_and_compare(
        X_train,
        y_train_high_cost,
        zoo=clf_zoo,
        cv=cv_clf,
        scoring=clf_scorers,
        groups=df_train["member_id"],
        experiment="member_high_cost_classification",
        primary_metric="average_precision",
    )
    logger.info(
        "Member High-Cost Classification Leaderboard:\n%s",
        clf_leaderboard[["model", "average_precision_mean", "roc_auc_mean"]],
    )

    # Tune top classifier
    best_clf_name = [m for m in clf_leaderboard["model"].tolist() if m != "dummy"][0]
    tuned_clf, _ = tune_model(
        name=f"member_clf_{best_clf_name}",
        base_estimator=clf_zoo[best_clf_name],
        X=X_train,
        y=y_train_high_cost,
        cv=cv_clf,
        scoring_metric="average_precision",
        n_trials=n_trials,
        seed=seed,
        groups=df_train["member_id"],
    )

    # Calibrate classifier on train split only
    calibrated_clf = CalibratedClassifierCV(estimator=tuned_clf, method="isotonic", cv=3)
    calibrated_clf.fit(X_train, y_train_high_cost)

    # 7. Establish Risk Tier Cutoffs from training expected cost predictions
    train_pred_costs = tuned_reg.predict(X_train)
    tier_quantiles = cfg.member.tier_quantiles  # [0.50, 0.80, 0.95]
    tier_cutoffs = [float(np.quantile(train_pred_costs, q)) for q in tier_quantiles]
    logger.info("Established member risk tier cutoffs from train predictions: %s", tier_cutoffs)

    # 8. Single Final Evaluation on Test Set
    test_pred_costs = tuned_reg.predict(X_test)
    test_pred_probs = calibrated_clf.predict_proba(X_test)[:, 1]

    reg_metrics = evaluate(y_test_cost, test_pred_costs, task="regression")
    clf_metrics = evaluate(
        y_test_high_cost,
        test_pred_probs,
        task="classification",
        y_prob=test_pred_probs,
        threshold=0.5,
    )

    combined_metrics = {
        "regression": reg_metrics,
        "classification": clf_metrics,
        "high_cost_threshold": round(high_cost_threshold, 2),
        "tier_cutoffs": [round(c, 2) for c in tier_cutoffs],
        "tier_quantiles": tier_quantiles,
        "n_train": len(df_train),
        "n_test": len(df_test),
        "test_r2": reg_metrics["r2"],
        "test_rmse": reg_metrics["rmse"],
        "pr_auc": clf_metrics["pr_auc"],
        "roc_auc": clf_metrics["roc_auc"],
        "f1": clf_metrics["f1"],
    }

    # Save metrics JSON
    metrics_path = cfg.paths.reports_dir / "metrics" / "member.json"
    save_metrics(combined_metrics, metrics_path)

    # 9. Figures: Calibration curve & PR curve
    fig_dir = cfg.paths.reports_dir / "figures"
    plot_calibration_curve(
        {"y_true": y_test_high_cost, "probs": test_pred_probs},
        fig_dir / "member_calibration.png",
    )
    plot_pr_curve(
        {"y_true": y_test_high_cost, "probs": test_pred_probs},
        fig_dir / "member_pr_curve.png",
    )

    # 10. Fairness Audit on Test Split
    sensitive_df = X_test.copy()
    if "age_band" not in sensitive_df.columns:
        sensitive_df["age_band"] = pd.cut(
            sensitive_df["age_at_year_end"],
            bins=[0, 65, 75, 85, 120],
            labels=["<65", "65-74", "75-84", "85+"],
        ).astype(str)

    fairness_results = fairness_audit(
        y_true=y_test_high_cost,
        y_pred=(test_pred_probs >= 0.5).astype(int),
        sensitive=sensitive_df[["sex", "race", "age_band"]],
        task="classification",
        y_prob=test_pred_probs,
        threshold=0.5,
        relative_gap_threshold=cfg.fairness.relative_gap_threshold,
        module="member",
    )

    # 11. Save model bundle
    now_utc = datetime.now(timezone.utc)
    version = now_utc.strftime("%Y%m%d_%H%M%S")
    bundle = {
        "regressor": tuned_reg,
        "classifier": calibrated_clf,
    }
    metadata = {
        "module": "member",
        "version": version,
        "features": list(X_train.columns),
        "training_rows": int(len(df_train)),
        "seed": seed,
        "metrics": combined_metrics,
        "high_cost_threshold": high_cost_threshold,
        "tier_cutoffs": tier_cutoffs,
        "tier_quantiles": tier_quantiles,
        "split_strategy": "GroupShuffleSplit (80/20) on member_id",
        "numeric_ranges": {
            col: {"min": float(X_train[col].min()), "max": float(X_train[col].max())}
            for col in num_cols
            if col in X_train.columns
        },
    }
    save_model("member", bundle, metadata)

    # 12. Create demo sample for HF / Streamlit deployment (Section 17)
    demo_dir = cfg.paths.raw_dir.parent / "demo"
    demo_dir.mkdir(parents=True, exist_ok=True)
    demo_sample = df_test.head(100).copy()
    demo_sample.to_parquet(demo_dir / "member_sample.parquet", index=False)

    # 13. Model Card
    generate_model_card("member")

    return {
        "bundle": bundle,
        "metadata": metadata,
        "metrics": combined_metrics,
        "fairness_results": fairness_results,
    }
