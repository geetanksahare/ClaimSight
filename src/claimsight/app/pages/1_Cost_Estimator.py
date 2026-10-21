import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from claimsight.explain.explain_prediction import explain_prediction
from claimsight.inference.predict_cost import predict_cost
from claimsight.inference.what_if import what_if, what_if_sweep

st.set_page_config(page_title="Cost Estimator | ClaimSight", page_icon="💰", layout="wide")

st.title("Individual Medical Cost Estimator")
st.info("Built on public/synthetic data. Not for real pricing, underwriting or claims decisions.")

try:
    # Test loading model
    from claimsight.models.registry import load_model

    load_model("cost")
except Exception as e:
    st.error(f"Cost model artifact not found or failed to load ({e}).")
    st.code("claimsight train-cost", language="bash")
    st.stop()

col_form, col_res = st.columns([1, 1.2])

with col_form:
    st.subheader("Member Attributes")
    with st.form("cost_form"):
        age_input_mode = st.radio(
            "Age Selection Mode",
            ["Standard Slider (18-64)", "Extended Input (18-100)"],
            horizontal=True,
        )
        if "Standard" in age_input_mode:
            age = st.slider("Age", min_value=18, max_value=64, value=35)
        else:
            age = st.number_input("Age (Extended Range)", min_value=18, max_value=100, value=35)

        sex = st.selectbox("Biological Sex", ["male", "female"], index=0)
        bmi = st.slider(
            "Body Mass Index (BMI)", min_value=15.0, max_value=55.0, value=28.5, step=0.1
        )
        children = st.slider("Dependent Children", min_value=0, max_value=5, value=1)
        smoker = st.selectbox("Smoking Status", ["no", "yes"], index=0)
        region = st.selectbox(
            "Geographic Region",
            ["northeast", "northwest", "southeast", "southwest"],
            index=1,
        )

        submitted = st.form_submit_button("Estimate Annual Charges", type="primary")

inputs = {
    "age": int(age),
    "sex": sex,
    "bmi": float(bmi),
    "children": int(children),
    "smoker": smoker,
    "region": region,
}

with col_res:
    st.subheader("Estimate & Uncertainty")
    pred_res = predict_cost(inputs)
    prediction = pred_res["prediction"]
    lower = pred_res["lower"]
    upper = pred_res["upper"]
    cov_level = int(pred_res["coverage_level"] * 100)

    m1, m2 = st.columns(2)
    m1.metric("Predicted Annual Charge", f"${prediction:,.2f}")
    m2.metric(f"{cov_level}% Conformal Interval", f"${lower:,.2f} – ${upper:,.2f}")

    if not pred_res["in_training_range"]:
        for w in pred_res["warnings"]:
            st.warning(w)

    # Position in dataset distribution gauge
    # Kaggle dataset percentiles roughly: 25th: $4,740, 50th: $9,382, 75th: $16,640, 90th: $34,800
    fig_gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prediction,
            number={"prefix": "$"},
            title={"text": "Relative Cost Position"},
            gauge={
                "axis": {"range": [0, 65000]},
                "bar": {"color": "#1f77b4"},
                "steps": [
                    {"range": [0, 9382], "color": "#e8f5e9"},
                    {"range": [9382, 16640], "color": "#fff9c4"},
                    {"range": [16640, 34800], "color": "#ffe0b2"},
                    {"range": [34800, 65000], "color": "#ffcdd2"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 3},
                    "thickness": 0.75,
                    "value": prediction,
                },
            },
        )
    )
    fig_gauge.update_layout(height=240, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig_gauge, use_container_width=True)

st.divider()

# Explainability & What-If Sections
exp_col, whatif_col = st.columns([1.1, 0.9])

with exp_col:
    st.subheader("Key Drivers (SHAP Local Explanation)")
    try:
        explanation = explain_prediction("cost", inputs, top_k=5)
        st.write(explanation["summary_text"])

        contrib_df = [
            {
                "Feature": c["feature"],
                "Effect (%)": c["effect_pct"],
                "Direction": c["direction"],
                "Narrative": c["text"],
            }
            for c in explanation["contributions"]
        ]
        fig_shap = px.bar(
            contrib_df,
            x="Effect (%)",
            y="Feature",
            orientation="h",
            color="Direction",
            color_discrete_map={"increases": "#e15759", "decreases": "#4e79a7"},
            title="Local Feature Contribution (% Multiplier on Log-Scale Baseline)",
        )
        fig_shap.update_layout(yaxis={"categoryorder": "total ascending"}, height=300)
        st.plotly_chart(fig_shap, use_container_width=True)

        for c in explanation["contributions"]:
            st.markdown(f"- {c['text']}")
    except Exception as ex:
        st.info(f"SHAP explanation unavailable: {ex}")

with whatif_col:
    st.subheader("Interactive What-If Simulation")
    sweep_feat = st.selectbox(
        "Select Parameter to Sweep", ["bmi", "age", "children", "smoker"], index=0
    )

    if sweep_feat == "bmi":
        vals = [18.5, 22.0, 25.0, 28.0, 30.0, 35.0, 40.0, 45.0]
    elif sweep_feat == "age":
        vals = [20, 25, 30, 35, 40, 45, 50, 55, 60, 64]
    elif sweep_feat == "children":
        vals = [0, 1, 2, 3, 4, 5]
    else:
        vals = ["no", "yes"]

    sweep_df = what_if_sweep(inputs, sweep_feat, vals)
    fig_sweep = px.line(
        sweep_df,
        x="feature_value",
        y="prediction",
        markers=True,
        title=f"Sensitivity: Estimated Cost vs {sweep_feat.upper()}",
        labels={"feature_value": sweep_feat, "prediction": "Predicted ($)"},
    )
    st.plotly_chart(fig_sweep, use_container_width=True)

    with st.expander("Sequential Modification Counterfactual"):
        target_smoker = "yes" if inputs["smoker"] == "no" else "no"
        sim_changes = {"smoker": target_smoker, "bmi": min(inputs["bmi"] + 5.0, 60.0)}
        sim_res = what_if(inputs, sim_changes)
        st.write(
            f"Modifying smoking to `{target_smoker}` and BMI by +5.0 shifts estimate by: **${sim_res['delta_abs']:+,.2f}** ({sim_res['delta_pct']:+.1f}%)"
        )
        st.table(sim_res["steps"])
