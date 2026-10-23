import json
import sys
from pathlib import Path

# Ensure root src directory is in sys.path for Streamlit Cloud deployment
_src_root = Path(__file__).resolve().parent.parent.parent
if str(_src_root) not in sys.path:
    sys.path.insert(0, str(_src_root))

import streamlit as st  # noqa: E402

st.set_page_config(
    page_title="ClaimSight | Health Insurance Cost & Claims Intelligence",
    page_icon="🛡️",
    layout="wide",
)

st.title("ClaimSight: Cost & Claims Intelligence Platform")

st.info("Built on public/synthetic data. Not for real pricing, underwriting or claims decisions.")

st.markdown("""
ClaimSight is a production-grade machine learning platform for health insurance cost prediction,
longitudinal member risk stratification, and provider fraud anomaly detection.
""")

reports_dir = Path("reports/metrics")
cost_metrics = {}
if (reports_dir / "cost.json").exists():
    with open(reports_dir / "cost.json", "r", encoding="utf-8") as f:
        cost_metrics = json.load(f)

col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("1. Cost Estimation")
    st.write(
        "Predicts individual annual medical charges with calibrated conformal prediction intervals."
    )
    if cost_metrics:
        st.metric("Test R²", f"{cost_metrics.get('r2', 0.0):.4f}")
        st.metric("Test RMSE", f"${cost_metrics.get('rmse', 0.0):,.2f}")
        st.metric("Interval Coverage", f"{cost_metrics.get('conformal_coverage', 0.0) * 100:.1f}%")
    else:
        st.caption("Model not yet trained. Run `claimsight train-cost`.")

with col2:
    st.subheader("2. Member Risk Stratification")
    st.write(
        "Predicts next-year expected cost and identifies high-risk members from prior claims history."
    )
    if (reports_dir / "member.json").exists():
        with open(reports_dir / "member.json", "r", encoding="utf-8") as f:
            m_metrics = json.load(f)
        st.metric("PR-AUC (High Cost)", f"{m_metrics.get('pr_auc', 0.0):.4f}")
    else:
        st.caption("Model not yet trained. Run `claimsight train-member`.")

with col3:
    st.subheader("3. Provider Anomaly Detection")
    st.write(
        "Blends supervised fraud prediction with Isolation Forest unsupervised anomaly scoring."
    )
    if (reports_dir / "fraud.json").exists():
        with open(reports_dir / "fraud.json", "r", encoding="utf-8") as f:
            f_metrics = json.load(f)
        st.metric("PR-AUC", f"{f_metrics.get('pr_auc', 0.0):.4f}")
    else:
        st.caption("Model not yet trained. Run `claimsight train-fraud`.")

st.divider()
st.subheader("Platform Modules")
st.markdown("""
Use the sidebar navigation to explore the platform:
- **Cost Estimator:** Interactive member cost prediction with SHAP contribution explanations and what-if counterfactual sweeps.
- **Portfolio Dashboard:** Population cohort drill-down, distribution analysis, and spending concentration.
- **Member Risk:** Multi-year Medicare SynPUF member expected cost and risk tier scoring.
- **Anomaly Review:** Healthcare provider fraud likelihood ranking, claim review inspector, and triage exports.
- **Batch Scoring:** Bulk CSV scoring with Pandera data schema validation and error feedback.
- **Model Performance:** Cross-validation leaderboards, test evaluation metrics, and diagnostic plots.
- **Fairness & Limitations:** Demographic disparity audits and model card documentation.
""")
