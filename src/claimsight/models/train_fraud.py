from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

from claimsight.config import get_settings
from claimsight.data.load_and_clean import load_and_clean
from claimsight.evaluation.evaluate import evaluate, save_metrics
from claimsight.evaluation.fairness_audit import fairness_audit
from claimsight.evaluation.plots import plot_pr_curve, plot_roc_curve
from claimsight.features.build_features import build_features
from claimsight.features.provider_features import split_xy
from claimsight.logging_config import get_logger
from claimsight.models.registry import save_model
from claimsight.models.train_and_compare import (
    get_default_classification_scorers,
    train_and_compare,
)
from claimsight.models.tune import tune_model
from claimsight.models.zoo import get_model_zoo
from claimsight.reporting.model_card import generate_model_card

logger = get_logger(__name__)


def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray, metric: str = "f1") -> float:
    """Find decision threshold maximizing F1 score on out-of-fold predictions."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    if metric == "f1":
        denom = precisions + recalls
        f1_scores = np.divide(
            2 * precisions * recalls,
            denom,
            out=np.zeros_like(precisions),
            where=denom > 0,
        )
        best_idx = int(np.argmax(f1_scores[:-1])) if len(thresholds) > 0 else 0
        best_threshold = float(thresholds[best_idx]) if len(thresholds) > 0 else 0.5
        logger.info(
            "Optimal out-of-fold %s threshold: %.4f (F1: %.4f)",
            metric,
            best_threshold,
            f1_scores[best_idx],
        )
        return best_threshold
    return 0.5


def train_fraud_pipeline(
    trials: int | None = None,
) -> dict[str, Any]:
    """Execute end-to-end provider anomaly and fraud risk scoring pipeline."""
    cfg = get_settings()
    seed = cfg.seed

    # 1. Load fraud data tables
    processed_dir = cfg.paths.processed_dir
    labels_file = processed_dir / "fraud_labels.parquet"
    if not labels_file.exists():
        fraud_data = load_and_clean("fraud", save=True)
    else:
        fraud_data = {
            "beneficiary": pd.read_parquet(processed_dir / "fraud_beneficiary.parquet"),
            "inpatient": pd.read_parquet(processed_dir / "fraud_inpatient.parquet"),
            "outpatient": pd.read_parquet(processed_dir / "fraud_outpatient.parquet"),
            "labels": pd.read_parquet(labels_file),
        }

    # 2. Build provider features
    df_feat = build_features("fraud", fraud_data, cfg)

    # 3. Stratified split 80/20 on is_fraud
    df_train, df_test = train_test_split(
        df_feat,
        test_size=cfg.fraud.test_size,
        stratify=df_feat["is_fraud"],
        random_state=seed,
    )
    logger.info(
        "Provider splits created - Train: %d (Fraud rate: %.2f%%), Test: %d (Fraud rate: %.2f%%)",
        len(df_train),
        df_train["is_fraud"].mean() * 100,
        len(df_test),
        df_test["is_fraud"].mean() * 100,
    )

    X_train_full, y_train = split_xy(df_train)
    X_test_full, y_test = split_xy(df_test)

    # Provider column is ID only, drop from modeling features
    X_train = X_train_full.drop(columns=["Provider"]).copy()
    X_test = X_test_full.drop(columns=["Provider"]).copy()

    # 4. Supervised Model Zoo & Leaderboard
    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)
    scale_pos = float(n_neg / max(n_pos, 1))

    num_cols = list(X_train.select_dtypes(include=[np.number]).columns)
    cat_cols = [c for c in X_train.columns if c not in num_cols]

    zoo = get_model_zoo(
        "fraud", seed=seed, scale_pos_weight=scale_pos, num_cols=num_cols, cat_cols=cat_cols
    )
    cv = StratifiedKFold(n_splits=cfg.fraud.cv_folds, shuffle=True, random_state=seed)
    scorers = get_default_classification_scorers()

    leaderboard = train_and_compare(
        X_train,
        y_train,
        zoo=zoo,
        cv=cv,
        scoring=scorers,
        experiment="provider_fraud_classification",
        primary_metric="average_precision",
    )
    logger.info(
        "Fraud Supervised Leaderboard:\n%s",
        leaderboard[["model", "average_precision_mean", "roc_auc_mean", "f1_mean"]],
    )

    # 5. Tune best classifier
    n_trials = trials if trials is not None else cfg.fraud.optuna_trials
    best_clf_name = [m for m in leaderboard["model"].tolist() if m != "dummy"][0]
    tuned_clf, _ = tune_model(
        name=f"fraud_{best_clf_name}",
        base_estimator=zoo[best_clf_name],
        X=X_train,
        y=y_train,
        cv=cv,
        scoring_metric="average_precision",
        n_trials=n_trials,
        seed=seed,
    )

    # 6. Pick decision threshold on out-of-fold predictions only
    oof_probs = cross_val_predict(
        tuned_clf,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]

    optimal_threshold = find_optimal_threshold(
        y_true=y_train.to_numpy(),
        y_prob=oof_probs,
        metric=cfg.fraud.threshold_metric,
    )

    # Refit tuned supervised model on full train
    tuned_clf.fit(X_train, y_train)

    # 7. Unsupervised Isolation Forest Anomaly Model
    # Fit strictly on train features without using labels
    iforest = IsolationForest(
        n_estimators=300,
        contamination=cfg.fraud.iforest_contamination,
        random_state=seed,
        n_jobs=-1,
    )
    # IsolationForest operates on numeric columns
    from sklearn.compose import ColumnTransformer

    ct = ColumnTransformer([("num", "passthrough", num_cols)], remainder="drop")
    X_train_num = ct.fit_transform(X_train)
    iforest.fit(X_train_num)

    train_decisions = iforest.decision_function(X_train_num)
    train_neg_scores = -train_decisions
    iforest_min = float(np.min(train_neg_scores))
    iforest_max = float(np.max(train_neg_scores))

    logger.info(
        "Isolation Forest fitted on train without labels: score bounds [%.4f, %.4f]",
        iforest_min,
        iforest_max,
    )

    # 8. Single Final Evaluation on Test Set
    test_probs = tuned_clf.predict_proba(X_test)[:, 1]
    base_rate = float(y_test.mean())

    test_metrics = evaluate(
        y_true=y_test,
        y_pred=test_probs,
        task="classification",
        y_prob=test_probs,
        threshold=optimal_threshold,
    )
    test_metrics["positive_base_rate"] = round(base_rate, 4)
    test_metrics["pr_auc_vs_base_rate_ratio"] = round(
        test_metrics["pr_auc"] / max(base_rate, 1e-4), 2
    )
    test_metrics["best_model"] = best_clf_name

    logger.info(
        "Final Test Evaluation: PR-AUC = %.4f (Base rate: %.4f, %.1fx lift), ROC-AUC = %.4f, F1 = %.4f at threshold %.4f",
        test_metrics["pr_auc"],
        base_rate,
        test_metrics["pr_auc_vs_base_rate_ratio"],
        test_metrics["roc_auc"],
        test_metrics["f1"],
        optimal_threshold,
    )

    # Save metrics JSON
    metrics_path = cfg.paths.reports_dir / "metrics" / "fraud.json"
    save_metrics(test_metrics, metrics_path)

    # 9. Figures
    fig_dir = cfg.paths.reports_dir / "figures"
    plot_pr_curve(
        {"y_true": y_test, "probs": test_probs},
        fig_dir / "fraud_pr_curve.png",
    )
    plot_roc_curve(
        {"y_true": y_test, "probs": test_probs},
        fig_dir / "fraud_roc_curve.png",
    )

    # 10. Fairness Audit
    # Since state is not provider-level, audit by inpatient_share terciles and provider size terciles
    sensitive_fraud = pd.DataFrame(index=X_test.index)
    sensitive_fraud["inpatient_share_tercile"] = pd.qcut(
        X_test["inpatient_share"], q=3, labels=["low", "medium", "high"], duplicates="drop"
    ).astype(str)
    sensitive_fraud["provider_size_tercile"] = pd.qcut(
        X_test["n_claims"], q=3, labels=["small", "medium", "large"], duplicates="drop"
    ).astype(str)

    fairness_results = fairness_audit(
        y_true=y_test,
        y_pred=(test_probs >= optimal_threshold).astype(int),
        sensitive=sensitive_fraud,
        task="classification",
        y_prob=test_probs,
        threshold=optimal_threshold,
        relative_gap_threshold=cfg.fairness.relative_gap_threshold,
        module="fraud",
    )

    # 11. Save Model Bundle
    now_utc = datetime.now(timezone.utc)
    version = now_utc.strftime("%Y%m%d_%H%M%S")
    bundle = {
        "classifier": tuned_clf,
        "isolation_forest": iforest,
        "column_transformer": ct,
    }
    metadata = {
        "module": "fraud",
        "version": version,
        "features": list(X_train.columns),
        "training_rows": int(len(X_train)),
        "seed": seed,
        "metrics": test_metrics,
        "optimal_threshold": float(optimal_threshold),
        "iforest_min": iforest_min,
        "iforest_max": iforest_max,
        "risk_weights": cfg.fraud.risk_weights.model_dump(),
        "split_strategy": "StratifiedKFold (80/20) on is_fraud",
    }
    save_model("fraud", bundle, metadata)

    # 12. Save held-out demo sample for deployment (Section 17)
    demo_dir = cfg.paths.raw_dir.parent / "demo"
    demo_dir.mkdir(parents=True, exist_ok=True)
    df_test_full = df_test.head(100).copy()
    df_test_full.to_parquet(demo_dir / "provider_sample.parquet", index=False)

    # 13. Model Card
    generate_model_card("fraud")

    return {
        "bundle": bundle,
        "metadata": metadata,
        "test_metrics": test_metrics,
        "fairness_results": fairness_results,
    }
