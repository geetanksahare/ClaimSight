from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from claimsight.config import get_settings
from claimsight.explain.explain_prediction import explain_prediction
from claimsight.models.registry import load_model


@lru_cache(maxsize=1)
def _get_fraud_bundle() -> tuple[dict[str, Any], dict[str, Any]]:
    return load_model("fraud")


def flag_anomalies(
    provider_features: pd.DataFrame,
    top_n: int | None = None,
    threshold: float | None = None,
) -> pd.DataFrame:
    """Rank healthcare providers by composite risk blending supervised fraud probability and unsupervised anomaly score."""
    cfg = get_settings()
    bundle, metadata = _get_fraud_bundle()

    classifier = bundle["classifier"]
    iforest = bundle["isolation_forest"]
    ct = bundle["column_transformer"]

    stored_thresh = float(metadata.get("optimal_threshold", 0.5))
    decision_threshold = threshold if threshold is not None else stored_thresh

    iforest_min = float(metadata.get("iforest_min", -0.2))
    iforest_max = float(metadata.get("iforest_max", 0.2))
    w_sup = cfg.fraud.risk_weights.supervised  # 0.7
    w_anom = cfg.fraud.risk_weights.anomaly  # 0.3

    df_in = provider_features.copy()
    has_labels = "is_fraud" in df_in.columns
    labels_series = df_in["is_fraud"].copy() if has_labels else None

    # Prepare features: exclude Provider and is_fraud
    drop_cols = [c for c in ["Provider", "is_fraud"] if c in df_in.columns]
    X = df_in.drop(columns=drop_cols)

    # 1. Supervised probability
    fraud_probs = classifier.predict_proba(X)[:, 1]

    # 2. Unsupervised Isolation Forest Anomaly Score
    X_num = ct.transform(X)
    decisions = iforest.decision_function(X_num)
    neg_scores = -decisions
    # Min-max normalization
    denom = max(iforest_max - iforest_min, 1e-6)
    anomaly_scores = np.clip((neg_scores - iforest_min) / denom, 0.0, 1.0)

    # 3. Composite Risk Score
    risk_scores = w_sup * fraud_probs + w_anom * anomaly_scores
    flagged = fraud_probs >= decision_threshold

    # Provider IDs
    provider_ids = (
        df_in["Provider"].tolist()
        if "Provider" in df_in.columns
        else [f"PRV_{i:04d}" for i in range(len(df_in))]
    )

    out_df = pd.DataFrame(
        {
            "provider": provider_ids,
            "fraud_prob": np.round(fraud_probs, 4),
            "anomaly_score": np.round(anomaly_scores, 4),
            "risk_score": np.round(risk_scores, 4),
            "flagged": flagged,
            "n_claims": df_in["n_claims"].tolist()
            if "n_claims" in df_in.columns
            else [0] * len(df_in),
            "reimb_sum": df_in["reimb_sum"].tolist()
            if "reimb_sum" in df_in.columns
            else [0.0] * len(df_in),
        }
    )

    if has_labels:
        out_df["label_for_review"] = labels_series.values

    # Sort descending by risk score
    out_df = out_df.sort_values(by="risk_score", ascending=False).reset_index(drop=True)
    out_df["rank"] = np.arange(1, len(out_df) + 1)

    # Limit to top_n if requested
    if top_n is not None and top_n > 0:
        out_df = out_df.head(top_n).copy()

    # Compute top reasons for each provider row
    top_reasons_list = []
    # Index X matching sorted out_df
    for i, row in out_df.iterrows():
        # Get matching X row
        prv_id = row["provider"]
        if "Provider" in df_in.columns:
            orig_idx = df_in[df_in["Provider"] == prv_id].index[0]
            x_row = X.loc[orig_idx:orig_idx]
        else:
            x_row = X.iloc[i : i + 1]

        try:
            exp = explain_prediction("fraud", x_row, top_k=3, model_bundle=(bundle, metadata))
            reasons = [c["text"] for c in exp["contributions"]]
        except Exception:
            reasons = [
                f"Elevated reimbursement volume (${row['reimb_sum']:,.2f})",
                f"Supervised risk probability ({row['fraud_prob'] * 100:.1f}%)",
                f"Isolation forest outlier score ({row['anomaly_score']:.2f})",
            ]
        top_reasons_list.append(reasons)

    out_df["top_reasons"] = top_reasons_list

    # Reorder columns as specified: provider, fraud_prob, anomaly_score, risk_score, flagged, rank, top_reasons, n_claims, reimb_sum
    cols_order = [
        "provider",
        "fraud_prob",
        "anomaly_score",
        "risk_score",
        "flagged",
        "rank",
        "top_reasons",
        "n_claims",
        "reimb_sum",
    ]
    if has_labels:
        cols_order.append("label_for_review")

    return out_df[cols_order]


def provider_claims(provider_id: str, claims_df: pd.DataFrame) -> pd.DataFrame:
    """Retrieve and rank all individual claims billed by a specific provider."""
    sub = claims_df[claims_df["Provider"] == provider_id].copy()
    if "InscClaimAmtReimbursed" in sub.columns:
        sub = sub.sort_values(by="InscClaimAmtReimbursed", ascending=False)
    return sub.reset_index(drop=True)
