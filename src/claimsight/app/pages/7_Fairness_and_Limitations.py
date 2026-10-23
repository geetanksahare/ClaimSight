import json
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Fairness & Limitations | ClaimSight", page_icon="⚖️", layout="wide")

st.title("⚖️ Demographic Disparity Auditing & System Limitations")

st.info("Built on public/synthetic data. Not for real pricing, underwriting or claims decisions.")

st.markdown("""
ClaimSight enforces quantitative demographic disparity audits powered by **Fairlearn** to evaluate
whether algorithmic performance diverges across protected or clinically sensitive cohorts.
The standard governance threshold is **≤ 20% relative performance gap** between subgroup extremes.
""")

tab_cost, tab_member, tab_fraud, tab_cards = st.tabs(
    [
        "1. Cost Model Fairness",
        "2. Member Risk Fairness",
        "3. Provider Anomaly Fairness",
        "4. Governance Model Cards",
    ]
)

metrics_dir = Path("reports/metrics")
cards_dir = Path("reports/model_cards")


def load_fairness(filename: str) -> dict:
    p = metrics_dir / filename
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


# 1. Cost Module Fairness
with tab_cost:
    st.header("Cost Prediction Model: Demographic Parity Audit")
    f_cost = load_fairness("fairness_cost.json")
    if not f_cost:
        st.warning("Fairness report for cost not found. Run `claimsight train-cost`.")
    else:
        st.markdown(
            f"**Audit Rule:** Maximum acceptable relative disparity is **{f_cost.get('relative_gap_threshold', 0.2) * 100:.0f}%**."
        )

        cols = f_cost.get("columns", {})
        for col_name, audit in cols.items():
            st.subheader(f"Audited Feature: `{col_name}`")
            is_disparate = audit.get("disparity_flag", False)
            rel_gap = audit.get("relative_gap", 0.0) * 100.0
            max_min = audit.get("max_min_gap", 0.0)

            if is_disparate:
                st.error(
                    f"⚠️ **Disparity Flag Triggered:** Relative gap is **{rel_gap:.1f}%** (exceeds 20% threshold). Max-Min MAE Gap: ${max_min:,.2f}."
                )
            else:
                st.success(
                    f"✅ **Fairness Passed:** Relative gap is **{rel_gap:.1f}%** (within 20% threshold)."
                )

            groups = audit.get("groups", {})
            rows = []
            for g_name, g_data in groups.items():
                rows.append(
                    {
                        "Subgroup": g_name,
                        "Sample Size (N)": int(g_data.get("count", 0)),
                        "MAE ($)": f"${g_data.get('mae', 0.0):,.2f}",
                        "R²": f"{g_data.get('r2', 0.0):.4f}",
                        "Selection / P90 Rate": f"{g_data.get('selection_rate', 0.0) * 100:.1f}%"
                        if "selection_rate" in g_data
                        else "N/A",
                        "Low Sample Alert": "⚠️ Yes (N<30)" if g_data.get("low_n", False) else "No",
                    }
                )
            st.dataframe(pd.DataFrame(rows), width="stretch")


# 2. Member Risk Fairness
with tab_member:
    st.header("Member Risk Model: Demographic Disparity Audit")
    f_member = load_fairness("fairness_member.json")
    if not f_member:
        st.warning("Fairness report for member risk not found. Run `claimsight train-member`.")
    else:
        cols = f_member.get("columns", {})
        for col_name, audit in cols.items():
            st.subheader(f"Audited Feature: `{col_name}`")
            is_disparate = audit.get("disparity_flag", False)
            rel_gap = audit.get("relative_gap", 0.0) * 100.0
            max_min = audit.get("max_min_gap", 0.0)

            if is_disparate:
                st.error(
                    f"⚠️ **Disparity Flag Triggered:** Relative gap is **{rel_gap:.1f}%** (exceeds 20% threshold). Selection Rate Gap: {max_min:.3f}."
                )
            else:
                st.success(
                    f"✅ **Fairness Passed:** Relative gap is **{rel_gap:.1f}%** (within 20% threshold)."
                )

            groups = audit.get("groups", {})
            rows = []
            for g_name, g_data in groups.items():
                rows.append(
                    {
                        "Subgroup": g_name,
                        "Sample Size (N)": int(g_data.get("count", 0)),
                        "Selection Rate": f"{g_data.get('selection_rate', 0.0) * 100:.1f}%",
                        "TPR (Recall)": f"{g_data.get('tpr', 0.0) * 100:.1f}%",
                        "FPR": f"{g_data.get('fpr', 0.0) * 100:.1f}%",
                        "Precision": f"{g_data.get('precision', 0.0) * 100:.1f}%",
                        "Low Sample Alert": "⚠️ Yes (N<30)" if g_data.get("low_n", False) else "No",
                    }
                )
            st.dataframe(pd.DataFrame(rows), width="stretch")


# 3. Provider Anomaly Fairness
with tab_fraud:
    st.header("Provider Anomaly Model: Provider Cohort Audit")
    f_fraud = load_fairness("fairness_fraud.json")
    if not f_fraud:
        st.warning("Fairness report for fraud not found. Run `claimsight train-fraud`.")
    else:
        cols = f_fraud.get("columns", {})
        for col_name, audit in cols.items():
            st.subheader(f"Audited Cohort: `{col_name}`")
            is_disparate = audit.get("disparity_flag", False)
            rel_gap = audit.get("relative_gap", 0.0) * 100.0

            if is_disparate:
                st.warning(
                    f"⚠️ **Auditing Notice:** Subgroup selection divergence is **{rel_gap:.1f}%**. "
                    f"Higher provider size or inpatient volume correlates naturally with audit prioritization."
                )
            else:
                st.success(f"✅ Disparity within expected bounds ({rel_gap:.1f}%).")

            groups = audit.get("groups", {})
            rows = []
            for g_name, g_data in groups.items():
                rows.append(
                    {
                        "Subgroup": g_name,
                        "Provider Count": int(g_data.get("count", 0)),
                        "Flag Rate": f"{g_data.get('selection_rate', 0.0) * 100:.1f}%",
                        "TPR": f"{g_data.get('tpr', 0.0) * 100:.1f}%",
                        "FPR": f"{g_data.get('fpr', 0.0) * 100:.1f}%",
                    }
                )
            st.dataframe(pd.DataFrame(rows), width="stretch")


# 4. Model Cards & Governance Documentation
with tab_cards:
    st.header("📜 Algorithmic Governance & Model Cards")
    st.markdown("""
    All models in ClaimSight are accompanied by standardized model cards documenting intended uses,
    training data sources, features, quantitative metrics, and explicit limitations.
    """)

    for module in ["cost", "member", "fraud"]:
        card_path = cards_dir / f"{module}.md"
        with st.expander(
            f"📄 Model Card: {module.title()} Intelligence Model", expanded=(module == "cost")
        ):
            if card_path.exists():
                st.markdown(card_path.read_text(encoding="utf-8"))
            else:
                st.warning(f"Model card for {module} not found.")

st.markdown("---")
st.subheader("⚠️ Clinical, Actuarial & Data Limitations")
st.markdown("""
1. **Public & Synthetic Datasets:**
   All models are trained strictly on public open-source or synthetically generated data (Kaggle Medical Cost, CMS SynPUF, Kaggle Medicare Fraud). No PHI or real commercial insurer proprietary claims are contained or utilized.
2. **Conformal Uncertainty:**
   Split conformal prediction intervals provide a statistical marginal coverage guarantee on exchangeable test distributions. Individual interval widths may vary and should be treated as bounds rather than absolute limits.
3. **No Automated Denial or Pricing:**
   Outputs generated by ClaimSight must never be used for real individual medical pricing, automated underwriting rejection, or automated denial of covered health benefits.
""")
