import json
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Model Performance | ClaimSight", page_icon="📈", layout="wide")

st.title("📈 Model Performance & Validation Observatory")

st.info("Built on public/synthetic data. Not for real pricing, underwriting or claims decisions.")

st.markdown("""
Comprehensive empirical evaluation metrics, validation curves, calibration profiles,
and conformal coverage audits across all three ClaimSight analytical modules.
Every metric is loaded dynamically from test set evaluation reports.
""")

tab_cost, tab_member, tab_fraud = st.tabs(
    [
        "1. Cost Prediction (Regression + Conformal)",
        "2. Member Risk Stratification (Tweedie + Isotonic)",
        "3. Provider Anomaly Review (Ensemble Supervised + IF)",
    ]
)

metrics_dir = Path("reports/metrics")
figures_dir = Path("reports/figures")


def load_json_metrics(filename: str) -> dict:
    p = metrics_dir / filename
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


# 1. Cost Module Tab
with tab_cost:
    st.header("Cost Estimation: Predictive Accuracy & Conformal Intervals")
    cost_metrics = load_json_metrics("cost.json")

    if not cost_metrics:
        st.warning("Cost metrics not found. Run `claimsight train-cost`.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Test R²", f"{cost_metrics.get('r2', 0.0):.4f}")
        with c2:
            st.metric("Test RMSE", f"${cost_metrics.get('rmse', 0.0):,.2f}")
        with c3:
            st.metric("Test MAE", f"${cost_metrics.get('mae', 0.0):,.2f}")
        with c4:
            cov = cost_metrics.get("conformal_coverage", 0.0) * 100.0
            st.metric("Conformal Coverage", f"{cov:.1f}%", "Target: ≥90%")

        st.markdown("### Actuarial Diagnostics & Coverage Validation")
        col_fig1, col_fig2 = st.columns(2)
        with col_fig1:
            p_actual = figures_dir / "cost_predicted_vs_actual.png"
            if p_actual.exists():
                st.image(str(p_actual), caption="Predicted vs. Actual Charges ($)")
        with col_fig2:
            p_res = figures_dir / "cost_residuals.png"
            if p_res.exists():
                st.image(str(p_res), caption="Residuals vs. Predicted Values")

        col_fig3, col_fig4 = st.columns(2)
        with col_fig3:
            p_cov = figures_dir / "cost_interval_coverage.png"
            if p_cov.exists():
                st.image(str(p_cov), caption="Empirical Conformal Interval Coverage")
        with col_fig4:
            p_smk = figures_dir / "cost_error_by_smoker.png"
            if p_smk.exists():
                st.image(str(p_smk), caption="Error Distribution by Smoking Subgroup")


# 2. Member Risk Tab
with tab_member:
    st.header("Member Longitudinal Risk: Tweedie Regression & Risk Tiers")
    member_metrics = load_json_metrics("member.json")

    if not member_metrics:
        st.warning("Member risk metrics not found. Run `claimsight train-member`.")
    else:
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("PR-AUC (High Cost)", f"{member_metrics.get('pr_auc', 0.0):.4f}")
        with m2:
            st.metric("ROC-AUC", f"{member_metrics.get('roc_auc', 0.0):.4f}")
        with m3:
            r2_val = member_metrics.get(
                "test_r2", member_metrics.get("regression", {}).get("r2", 0.0)
            )
            rmse_val = member_metrics.get(
                "test_rmse", member_metrics.get("regression", {}).get("rmse", 0.0)
            )
            st.metric("Test R²", f"{r2_val:.4f}", f"RMSE: ${rmse_val:,.2f}")
        with m4:
            brier = member_metrics.get("classification", {}).get(
                "brier", member_metrics.get("brier", 0.0)
            )
            st.metric("Brier Score", f"{brier:.4f}")

        st.markdown("### Precision-Recall & Calibration Profiles")
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            p_pr = figures_dir / "member_pr_curve.png"
            if p_pr.exists():
                st.image(str(p_pr), caption="Precision-Recall Curve vs Base Rate")
        with col_m2:
            p_cal = figures_dir / "member_calibration.png"
            if p_cal.exists():
                st.image(str(p_cal), caption="Isotonic Probability Calibration Curve")


# 3. Fraud Anomaly Review Tab
with tab_fraud:
    st.header("Provider Fraud Anomaly Detection: Supervised + Isolation Forest")
    fraud_metrics = load_json_metrics("fraud.json")

    if not fraud_metrics:
        st.warning("Fraud metrics not found. Run `claimsight train-fraud`.")
    else:
        f1, f2, f3, f4 = st.columns(4)
        with f1:
            st.metric("PR-AUC", f"{fraud_metrics.get('pr_auc', 0.0):.4f}")
        with f2:
            st.metric("ROC-AUC", f"{fraud_metrics.get('roc_auc', 0.0):.4f}")
        with f3:
            st.metric("Optimal F1 Score", f"{fraud_metrics.get('f1', 0.0):.4f}")
        with f4:
            thresh = fraud_metrics.get("threshold_used", 0.5)
            st.metric("Decision Threshold", f"{thresh:.3f}")

        st.markdown("### Receiver Operating Characteristic & Precision-Recall")
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            p_fpr = figures_dir / "fraud_pr_curve.png"
            if p_fpr.exists():
                st.image(str(p_fpr), caption="Provider Fraud Precision-Recall Curve")
        with col_f2:
            p_froc = figures_dir / "fraud_roc_curve.png"
            if p_froc.exists():
                st.image(str(p_froc), caption="Provider Fraud ROC Curve")

        st.markdown("### Interactive Decision Threshold Simulation")
        sim_thresh = st.slider(
            "Simulate Classification Decision Threshold", 0.1, 0.9, float(thresh), 0.05
        )
        st.caption(
            f"Simulating threshold at {sim_thresh:.2f}: balances audit capacity (precision) against missed potential non-compliance (recall)."
        )
