from pathlib import Path

import pandas as pd
import streamlit as st

from claimsight.inference.flag_anomalies import flag_anomalies, provider_claims

st.set_page_config(page_title="Anomaly Review | ClaimSight", page_icon="🔍", layout="wide")

st.title("Healthcare Provider Fraud & Anomaly Triage")
st.info("Built on public/synthetic data. Not for real pricing, underwriting or claims decisions.")

st.markdown(
    """
    > **Review Guidance:** Providers identified with elevated risk scores or flagged indicators are
    > prioritized for **human clinical & audit review**. A flag indicates statistical deviation and elevated
    > risk characteristics; it does **not** prove intentional fraud or misconduct.
    """
)

try:
    from claimsight.models.registry import load_model

    bundle, meta = load_model("fraud")
except Exception as e:
    st.error(f"Provider fraud model artifact not found or failed to load ({e}).")
    st.code("claimsight train-fraud", language="bash")
    st.stop()

# Load held-out test or demo provider features
demo_file = Path("data/demo/provider_sample.parquet")
if demo_file.exists():
    df_providers = pd.read_parquet(demo_file)
else:
    # Build from processed data
    from claimsight.config import get_settings
    from claimsight.features.build_features import build_features

    cfg = get_settings()
    processed_dir = cfg.paths.processed_dir
    if (processed_dir / "fraud_labels.parquet").exists():
        f_data = {
            "beneficiary": pd.read_parquet(processed_dir / "fraud_beneficiary.parquet"),
            "inpatient": pd.read_parquet(processed_dir / "fraud_inpatient.parquet"),
            "outpatient": pd.read_parquet(processed_dir / "fraud_outpatient.parquet"),
            "labels": pd.read_parquet(processed_dir / "fraud_labels.parquet"),
        }
        df_providers = build_features("fraud", f_data, cfg)
    else:
        st.error("No provider dataset found. Run `claimsight train-fraud`.")
        st.stop()

# Sidebar filters
st.sidebar.subheader("Review Filters")
min_risk = st.sidebar.slider("Minimum Composite Risk Score", 0.0, 1.0, 0.20, step=0.05)
flag_filter = st.sidebar.selectbox(
    "Flagged Status Filter", ["All Providers", "Flagged for Review Only", "Unflagged Only"]
)

# Score anomalies
flagged_df = flag_anomalies(df_providers)

# Apply filters
filtered_df = flagged_df[flagged_df["risk_score"] >= min_risk].copy()
if flag_filter == "Flagged for Review Only":
    filtered_df = filtered_df[filtered_df["flagged"] is True]
elif flag_filter == "Unflagged Only":
    filtered_df = filtered_df[filtered_df["flagged"] is False]

k1, k2, k3, k4 = st.columns(4)
k1.metric("Providers Scored", len(flagged_df))
k2.metric("Matching Filter", len(filtered_df))
k3.metric("Flagged for Audit", int(filtered_df["flagged"].sum()))
k4.metric(
    "Avg Composite Risk",
    f"{filtered_df['risk_score'].mean():.3f}" if len(filtered_df) > 0 else "0.000",
)

st.divider()

col_table, col_detail = st.columns([1.2, 0.8])

with col_table:
    st.subheader("Provider Risk Ranking")
    display_cols = [
        "rank",
        "provider",
        "risk_score",
        "fraud_prob",
        "anomaly_score",
        "flagged",
        "n_claims",
        "reimb_sum",
    ]
    if "label_for_review" in filtered_df.columns:
        display_cols.append("label_for_review")

    st.dataframe(
        filtered_df[display_cols].style.format(
            {
                "risk_score": "{:.3f}",
                "fraud_prob": "{:.3f}",
                "anomaly_score": "{:.3f}",
                "reimb_sum": "${:,.2f}",
            }
        ),
        height=400,
        width="stretch",
    )

    csv_data = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Review Triage Queue (CSV)",
        data=csv_data,
        file_name="claimsight_fraud_review_queue.csv",
        mime="text/csv",
    )

with col_detail:
    st.subheader("Provider Audit Inspector")
    if len(filtered_df) > 0:
        selected_prv = st.selectbox(
            "Select Provider for Deep Dive", filtered_df["provider"].tolist()
        )
        prv_row = filtered_df[filtered_df["provider"] == selected_prv].iloc[0]

        st.markdown(
            f"**Provider ID:** `{selected_prv}`  \n**Rank:** #{prv_row['rank']} | **Risk Score:** {prv_row['risk_score']:.3f}"
        )
        status_label = "🔴 FLAGGED FOR REVIEW" if prv_row["flagged"] else "🟢 NOT FLAGGED"
        st.markdown(f"**Audit Status:** {status_label}")

        st.write("**Top Algorithmic Drivers / Risk Reasons:**")
        for reason in prv_row["top_reasons"]:
            st.markdown(f"- {reason}")

        # Inspection of claims
        st.subheader("Top Billed Claims")
        raw_ip = Path("data/processed/fraud_inpatient.parquet")
        raw_op = Path("data/processed/fraud_outpatient.parquet")
        if raw_ip.exists() and raw_op.exists():
            all_c = pd.concat([pd.read_parquet(raw_ip), pd.read_parquet(raw_op)], ignore_index=True)
            claims_table = provider_claims(selected_prv, all_c)
            st.write(f"Total Claims on File: {len(claims_table)}")
            show_claim_cols = [
                c
                for c in [
                    "ClaimID",
                    "ClaimStartDt",
                    "InscClaimAmtReimbursed",
                    "DeductibleAmtPaid",
                    "claim_type",
                ]
                if c in claims_table.columns
            ]
            st.dataframe(claims_table[show_claim_cols].head(10), width="stretch")
        else:
            st.caption("Detailed claims table not loaded.")
    else:
        st.info("No providers match the current filter criteria.")
