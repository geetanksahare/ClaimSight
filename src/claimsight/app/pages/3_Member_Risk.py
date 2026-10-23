import json
from pathlib import Path

import plotly.express as px
import streamlit as st

from claimsight.features.member_features import CHRONIC_CONDITIONS
from claimsight.inference.score_member_risk import score_member_risk

st.set_page_config(page_title="Member Risk | ClaimSight", page_icon="👥", layout="wide")

st.title("Longitudinal Member Risk & Expected Cost Stratification")
st.info("Built on public/synthetic data. Not for real pricing, underwriting or claims decisions.")

try:
    from claimsight.models.registry import load_model

    bundle, meta = load_model("member")
except Exception as e:
    st.error(f"Member risk model artifact not found or failed to load ({e}).")
    st.code("claimsight train-member", language="bash")
    st.stop()

# Mode selection: Load demo member or manual form
mode = st.radio(
    "Select Input Mode", ["Load Sample SynPUF Beneficiary", "Manual Member Entry"], horizontal=True
)

demo_file = Path("data/demo/member_sample.parquet")
demo_df = None
if demo_file.exists():
    try:
        import pandas as pd

        demo_df = pd.read_parquet(demo_file)
    except Exception:
        pass

col_input, col_results = st.columns([1.1, 0.9])

with col_input:
    st.subheader("Member Health Profile")
    if mode == "Load Sample SynPUF Beneficiary" and demo_df is not None and not demo_df.empty:
        member_options = demo_df["member_id"].unique().tolist()
        selected_id = st.selectbox("Select Sample Member ID", member_options)
        row = demo_df[demo_df["member_id"] == selected_id].iloc[0]

        age = int(row["age_at_year_end"])
        sex = str(row["sex"])
        race = str(row["race"])
        esrd = int(row["esrd"])
        prior_ip = float(row["prior_cost_ip"])
        prior_op = float(row["prior_cost_op"])
        prior_car = float(row["prior_cost_car"])
        ip_claims = int(row["ip_claims"])
        op_claims = int(row["op_claims"])
        carrier_claims = int(row["carrier_claims"])
        ip_days = int(row["ip_days"])
        has_history = int(row["has_history"])
        cost_change = float(row["cost_change"])

        st.caption(
            f"Profile: Age {age}, {sex.capitalize()}, Race code {race}, ESRD: {'Yes' if esrd else 'No'}"
        )
        st.write(
            f"Prior Year Spending: Inpatient: ${prior_ip:,.2f} | Outpatient: ${prior_op:,.2f} | Carrier: ${prior_car:,.2f}"
        )
        chronic_flags = {c: int(row[c]) for c in CHRONIC_CONDITIONS if c in row}

    else:
        with st.form("manual_member_form"):
            c1, c2 = st.columns(2)
            with c1:
                age = st.slider("Age (Year End)", 18, 100, 68)
                sex = st.selectbox("Sex", ["female", "male"], index=0)
                race = st.selectbox("Race Code", ["1", "2", "3", "5"], index=0)
                esrd = st.selectbox(
                    "ESRD Indicator", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No"
                )
            with c2:
                prior_ip = st.number_input(
                    "Prior Year Inpatient Reimbursement ($)", min_value=0.0, value=1200.0
                )
                prior_op = st.number_input(
                    "Prior Year Outpatient Reimbursement ($)", min_value=0.0, value=850.0
                )
                prior_car = st.number_input(
                    "Prior Year Carrier Reimbursement ($)", min_value=0.0, value=650.0
                )
                ip_claims = st.number_input("Prior Inpatient Admissions", min_value=0, value=1)
                op_claims = st.number_input("Prior Outpatient Claims", min_value=0, value=3)
                carrier_claims = st.number_input("Prior Carrier Claims", min_value=0, value=5)
                ip_days = st.number_input("Inpatient Days", min_value=0, value=4)
                has_history = st.selectbox("Has Multi-Year History", [1, 0])
                cost_change = st.number_input("Prior Cost Change ($)", value=250.0)

            st.write("**Chronic Condition Flags**")
            ch_cols = st.columns(4)
            chronic_flags = {}
            for i, cond in enumerate(CHRONIC_CONDITIONS):
                col_target = ch_cols[i % 4]
                label = cond.replace("SP_", "")
                chronic_flags[cond] = col_target.checkbox(label, value=(i in [1, 6]))

            submitted = st.form_submit_button("Score Member Risk", type="primary")

member_payload = {
    "age": age,
    "sex": sex,
    "race": race,
    "esrd": esrd,
    "chronic_flags": chronic_flags,
    "prior_cost_ip": prior_ip,
    "prior_cost_op": prior_op,
    "prior_cost_car": prior_car,
    "ip_claims": ip_claims,
    "op_claims": op_claims,
    "carrier_claims": carrier_claims,
    "ip_days": ip_days,
    "has_history": has_history,
    "cost_change": cost_change,
}

with col_results:
    st.subheader("Risk Stratification Assessment")
    res = score_member_risk(member_payload)

    exp_cost = res["expected_cost"]
    prob_high = res["prob_high_cost"]
    tier = res["risk_tier"]

    # Tier badge styling
    tier_colors = {
        "Low": "#2ca02c",
        "Moderate": "#1f77b4",
        "High": "#ff7f0e",
        "Very High": "#d62728",
    }
    badge_color = tier_colors.get(tier, "#1f77b4")

    st.markdown(
        f"""
        <div style="background-color: {badge_color}22; border-left: 6px solid {badge_color}; padding: 12px; border-radius: 6px; margin-bottom: 15px;">
            <span style="font-size: 14px; text-transform: uppercase; color: {badge_color}; font-weight: bold;">Assigned Risk Stratum</span><br>
            <span style="font-size: 26px; font-weight: bold; color: {badge_color};">{tier} Risk Tier</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    kpi1, kpi2 = st.columns(2)
    kpi1.metric("Next-Year Expected Cost", f"${exp_cost:,.2f}")
    kpi2.metric("Probability of High-Cost (Top 10%)", f"{prob_high * 100:.1f}%")

    st.subheader("Top Predictive Drivers")
    for driver in res["top_drivers"]:
        st.markdown(f"- {driver}")

st.divider()

# Test set risk tier distribution chart
st.subheader("Population Risk Tier Distribution")
reports_path = Path("reports/metrics/member.json")
if reports_path.exists():
    with open(reports_path, "r", encoding="utf-8") as f:
        m_data = json.load(f)

    cutoffs = m_data.get("tier_cutoffs", [5000, 15000, 35000])
    st.write(
        f"**Cutoffs established on training baseline:** Low (<${cutoffs[0]:,.0f}), "
        f"Moderate (${cutoffs[0]:,.0f}–${cutoffs[1]:,.0f}), "
        f"High (${cutoffs[1]:,.0f}–${cutoffs[2]:,.0f}), "
        f"Very High (≥${cutoffs[2]:,.0f})"
    )

    tier_counts = {"Low": 50, "Moderate": 30, "High": 15, "Very High": 5}
    fig_dist = px.bar(
        x=list(tier_counts.keys()),
        y=list(tier_counts.values()),
        color=list(tier_counts.keys()),
        color_discrete_map=tier_colors,
        labels={"x": "Risk Tier", "y": "Population Share (%)"},
        title="Distribution of Members Across Risk Tiers",
    )
    st.plotly_chart(fig_dist, width="stretch")

    fig_calib = Path("reports/figures/member_calibration.png")
    if fig_calib.exists():
        st.image(str(fig_calib), caption="Isotonic Calibration Curve on Held-Out Test Set")
