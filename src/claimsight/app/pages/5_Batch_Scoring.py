import io

import pandas as pd
import streamlit as st

from claimsight.config import get_settings
from claimsight.inference.flag_anomalies import flag_anomalies
from claimsight.inference.predict_cost import predict_cost
from claimsight.inference.score_member_risk import score_member_risk

st.set_page_config(page_title="Batch Scoring | ClaimSight", page_icon="📁", layout="wide")

st.title("📁 Batch Scoring & Validation Engine")

st.info("Built on public/synthetic data. Not for real pricing, underwriting or claims decisions.")

st.markdown("""
Upload a batch dataset (CSV) to run high-throughput inference across any ClaimSight pipeline.
Input files are validated against strict actuarial and clinical schemas before model scoring.
""")

module_choice = st.radio(
    "Select Target Scoring Pipeline:",
    options=["Annual Cost Estimation", "Member Risk Stratification", "Provider Anomaly Review"],
    horizontal=True,
)


# Template generator helper
def get_template(choice: str) -> pd.DataFrame:
    if choice == "Annual Cost Estimation":
        return pd.DataFrame(
            [
                {
                    "age": 35,
                    "sex": "male",
                    "bmi": 26.5,
                    "children": 1,
                    "smoker": "no",
                    "region": "northwest",
                },
                {
                    "age": 52,
                    "sex": "female",
                    "bmi": 34.2,
                    "children": 0,
                    "smoker": "yes",
                    "region": "southeast",
                },
                {
                    "age": 28,
                    "sex": "female",
                    "bmi": 22.0,
                    "children": 2,
                    "smoker": "no",
                    "region": "northeast",
                },
            ]
        )
    elif choice == "Member Risk Stratification":
        return pd.DataFrame(
            [
                {
                    "member_id": "MEM_101",
                    "age": 67,
                    "sex": "female",
                    "race": "1",
                    "esrd": 0,
                    "prior_cost_ip": 1200.0,
                    "prior_cost_op": 500.0,
                    "prior_cost_car": 350.0,
                    "ip_claims": 1,
                    "op_claims": 3,
                    "carrier_claims": 5,
                    "ip_days": 2,
                    "has_history": 1,
                    "cost_change": 150.0,
                    "SP_DIABETES": 1,
                    "SP_CHF": 0,
                },
                {
                    "member_id": "MEM_102",
                    "age": 74,
                    "sex": "male",
                    "race": "1",
                    "esrd": 1,
                    "prior_cost_ip": 5400.0,
                    "prior_cost_op": 1200.0,
                    "prior_cost_car": 800.0,
                    "ip_claims": 3,
                    "op_claims": 6,
                    "carrier_claims": 12,
                    "ip_days": 8,
                    "has_history": 1,
                    "cost_change": 1200.0,
                    "SP_DIABETES": 1,
                    "SP_CHF": 1,
                },
            ]
        )
    else:
        cfg = get_settings()
        sample_path = cfg.paths.demo_dir / "provider_sample.parquet"
        if sample_path.exists():
            return pd.read_parquet(sample_path).head(3)
        return pd.DataFrame([{"Provider": "PRV_001", "n_claims": 10, "reimb_sum": 15000.0}])


# Template download button
template_df = get_template(module_choice)
csv_buffer = io.StringIO()
template_df.to_csv(csv_buffer, index=False)
st.download_button(
    label=f"⬇️ Download CSV Template for {module_choice}",
    data=csv_buffer.getvalue(),
    file_name=f"template_{module_choice.lower().replace(' ', '_')}.csv",
    mime="text/csv",
)

st.markdown("---")

uploaded_file = st.file_uploader("Upload CSV Data File for Scoring", type=["csv"])

if uploaded_file is not None:
    try:
        input_df = pd.read_csv(uploaded_file)
        st.success(
            f"Uploaded file loaded successfully: {len(input_df):,} rows, {len(input_df.columns)} columns."
        )
    except Exception as e:
        st.error(f"Error parsing CSV file: {e}")
        st.stop()

    errors = []

    # Schema Validation
    if module_choice == "Annual Cost Estimation":
        required_cols = ["age", "sex", "bmi", "children", "smoker", "region"]
        for col in required_cols:
            if col not in input_df.columns:
                errors.append(
                    {"row": "All", "column": col, "issue": f"Missing required column: '{col}'"}
                )

        if not errors:
            for idx, row in input_df.iterrows():
                try:
                    age = int(row["age"])
                    if not (18 <= age <= 100):
                        errors.append(
                            {
                                "row": idx + 1,
                                "column": "age",
                                "issue": f"Age {age} out of range [18, 100]",
                            }
                        )
                except Exception:
                    errors.append(
                        {"row": idx + 1, "column": "age", "issue": "Age must be numeric integer"}
                    )

                if str(row["sex"]).lower() not in ["male", "female"]:
                    errors.append(
                        {"row": idx + 1, "column": "sex", "issue": f"Invalid sex '{row['sex']}'"}
                    )

                try:
                    bmi = float(row["bmi"])
                    if not (10.0 <= bmi <= 70.0):
                        errors.append(
                            {
                                "row": idx + 1,
                                "column": "bmi",
                                "issue": f"BMI {bmi} out of range [10, 70]",
                            }
                        )
                except Exception:
                    errors.append(
                        {"row": idx + 1, "column": "bmi", "issue": "BMI must be numeric float"}
                    )

                if str(row["smoker"]).lower() not in ["yes", "no"]:
                    errors.append(
                        {
                            "row": idx + 1,
                            "column": "smoker",
                            "issue": f"Invalid smoker flag '{row['smoker']}'",
                        }
                    )

    elif module_choice == "Member Risk Stratification":
        required_cols = ["age", "sex"]
        for col in required_cols:
            if col not in input_df.columns:
                errors.append(
                    {"row": "All", "column": col, "issue": f"Missing required column: '{col}'"}
                )

    # Error handling display
    if errors:
        st.error(f"⚠️ Validation checks identified {len(errors)} error(s) in uploaded data:")
        err_df = pd.DataFrame(errors)
        st.dataframe(err_df, use_container_width=True)
        st.warning("Please correct these errors in your CSV file before proceeding.")
    else:
        st.success("✅ File passed schema validation! Ready for inference.")
        if st.button("🚀 Run Batch Scoring", type="primary"):
            with st.spinner("Scoring dataset..."):
                try:
                    if module_choice == "Annual Cost Estimation":
                        scored_df = predict_cost(input_df)
                    elif module_choice == "Member Risk Stratification":
                        scored_df = score_member_risk(input_df)
                    else:
                        scored_df = flag_anomalies(input_df)

                    st.balloons()
                    st.subheader("🎯 Batch Scoring Results")

                    m1, m2, m3 = st.columns(3)
                    with m1:
                        st.metric("Total Scored Records", f"{len(scored_df):,}")
                    with m2:
                        if "prediction" in scored_df.columns:
                            st.metric(
                                "Mean Predicted Cost", f"${scored_df['prediction'].mean():,.2f}"
                            )
                        elif "expected_cost" in scored_df.columns:
                            st.metric(
                                "Mean Prospective Cost",
                                f"${scored_df['expected_cost'].mean():,.2f}",
                            )
                        elif "risk_score" in scored_df.columns:
                            st.metric("Mean Risk Score", f"{scored_df['risk_score'].mean():.4f}")
                    with m3:
                        if "flagged" in scored_df.columns:
                            st.metric("Flagged Outliers", f"{scored_df['flagged'].sum():,}")
                        elif "risk_tier" in scored_df.columns:
                            high_count = (scored_df["risk_tier"] == "High").sum()
                            st.metric("High-Risk Members", f"{high_count:,}")
                        elif "prediction" in scored_df.columns:
                            p90 = scored_df["prediction"].quantile(0.9)
                            st.metric("90th Percentile Cost", f"${p90:,.2f}")

                    st.dataframe(scored_df, use_container_width=True)

                    out_csv = scored_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="📥 Download Scored Results (CSV)",
                        data=out_csv,
                        file_name=f"scored_{module_choice.lower().replace(' ', '_')}.csv",
                        mime="text/csv",
                    )
                except Exception as e:
                    st.error(f"Inference execution failed: {e}")
