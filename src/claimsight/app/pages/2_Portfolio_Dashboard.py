import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from claimsight.config import get_settings
from claimsight.inference.predict_cost import predict_cost

st.set_page_config(page_title="Portfolio Dashboard | ClaimSight", page_icon="📊", layout="wide")

st.title("📊 Portfolio Risk & Cost Dashboard")

st.info("Built on public/synthetic data. Not for real pricing, underwriting or claims decisions.")

st.markdown("""
Interactive actuarial population dashboard for analyzing portfolio risk exposure, demographic cost drivers,
and segment-level expenditure distributions.
""")


@st.cache_data
def load_portfolio_data() -> pd.DataFrame:
    cfg = get_settings()
    cost_path = cfg.paths.processed_dir / "cost.parquet"
    if cost_path.exists():
        df = pd.read_parquet(cost_path)
    else:
        # Fallback to demo cost sample
        demo_cost_path = cfg.paths.demo_dir / "insurance_clean.parquet"
        if demo_cost_path.exists():
            df = pd.read_parquet(demo_cost_path)
        else:
            st.error(
                "No portfolio data found. Run `claimsight load-and-clean` or provide `data/demo/insurance_clean.parquet`."
            )
            st.stop()

    if "charges" in df.columns and "prediction" not in df.columns:
        # Score the portfolio using the cost model
        features_df = df[["age", "sex", "bmi", "children", "smoker", "region"]]
        scored = predict_cost(features_df)
        df["predicted_cost"] = scored["prediction"].values
        df["interval_lower"] = scored["lower"].values
        df["interval_upper"] = scored["upper"].values
    return df


df = load_portfolio_data()

# Sidebar filters
st.sidebar.header("🔍 Cohort Filters")

age_min, age_max = int(df["age"].min()), int(df["age"].max())
selected_age = st.sidebar.slider(
    "Age Range", min_value=age_min, max_value=age_max, value=(age_min, age_max)
)

sex_options = sorted(df["sex"].unique())
selected_sex = st.sidebar.multiselect("Sex", options=sex_options, default=sex_options)

smoker_options = sorted(df["smoker"].unique())
selected_smoker = st.sidebar.multiselect(
    "Smoking Status", options=smoker_options, default=smoker_options
)

region_options = sorted(df["region"].unique())
selected_region = st.sidebar.multiselect(
    "Geographic Region", options=region_options, default=region_options
)

# Filter dataset
filtered = df[
    (df["age"] >= selected_age[0])
    & (df["age"] <= selected_age[1])
    & (df["sex"].isin(selected_sex))
    & (df["smoker"].isin(selected_smoker))
    & (df["region"].isin(selected_region))
].copy()

if filtered.empty:
    st.warning("No records match the selected filter combination. Please broaden your selection.")
    st.stop()

# Cost target series
cost_col = "predicted_cost" if "predicted_cost" in filtered.columns else "charges"
actual_col = "charges" if "charges" in filtered.columns else None

# Population KPIs
st.subheader("📈 Cohort Summary KPIs")
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

total_members = len(filtered)
total_spend = filtered[cost_col].sum()
mean_cost = filtered[cost_col].mean()
p90_cost = df[cost_col].quantile(0.90)
high_cost_count = (filtered[cost_col] >= p90_cost).sum()
high_cost_share = (high_cost_count / total_members) * 100.0

with kpi1:
    st.metric("Total Enrollees", f"{total_members:,}")
with kpi2:
    st.metric("Mean Predicted Cost", f"${mean_cost:,.2f}")
with kpi3:
    st.metric("Total Annual Spend", f"${total_spend:,.2f}")
with kpi4:
    st.metric(
        "High-Cost Outliers (≥90th)", f"{high_cost_count:,}", f"{high_cost_share:.1f}% of cohort"
    )

st.markdown("---")

# Visualizations
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Distribution of Predicted Annual Charges")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.histplot(filtered[cost_col], kde=True, color="#1e3a8a", ax=ax, bins=25)
    ax.axvline(
        mean_cost, color="#dc2626", linestyle="--", linewidth=1.5, label=f"Mean: ${mean_cost:,.0f}"
    )
    ax.axvline(
        filtered[cost_col].median(),
        color="#059669",
        linestyle=":",
        linewidth=1.5,
        label=f"Median: ${filtered[cost_col].median():,.0f}",
    )
    ax.set_xlabel("Predicted Medical Charges ($)")
    ax.set_ylabel("Enrollee Count")
    ax.set_title("Cost Distribution")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

with col_right:
    st.subheader("Risk Segment Heatmap: BMI vs Smoking")
    # Discretize BMI into standard clinical categories
    bmi_bins = [0, 18.5, 24.9, 29.9, 39.9, 100.0]
    bmi_labels = ["Underweight", "Normal", "Overweight", "Obese Class I/II", "Extreme Obesity"]
    filtered["bmi_category"] = pd.cut(
        filtered["bmi"], bins=bmi_bins, labels=bmi_labels, right=False
    )

    pivot_table = filtered.pivot_table(
        values=cost_col,
        index="bmi_category",
        columns="smoker",
        aggfunc="mean",
        observed=False,
    )

    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.heatmap(
        pivot_table,
        annot=True,
        fmt=",.0f",
        cmap="YlOrRd",
        cbar_kws={"label": "Mean Cost ($)"},
        ax=ax,
    )
    ax.set_title("Average Predicted Charges by BMI and Smoking Status")
    ax.set_xlabel("Smoking Status")
    ax.set_ylabel("Clinical BMI Category")
    st.pyplot(fig)
    plt.close(fig)

# Regional breakdown
st.subheader("Geographic & Demographic Spend Comparison")
col_reg1, col_reg2 = st.columns(2)

with col_reg1:
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(
        data=filtered,
        x="region",
        y=cost_col,
        hue="sex",
        palette="Blues",
        ax=ax,
        errorbar=None,
    )
    ax.set_title("Average Cost by Region and Sex")
    ax.set_xlabel("Region")
    ax.set_ylabel("Mean Charges ($)")
    st.pyplot(fig)
    plt.close(fig)

with col_reg2:
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.scatterplot(
        data=filtered,
        x="age",
        y=cost_col,
        hue="smoker",
        palette={"yes": "#dc2626", "no": "#2563eb"},
        alpha=0.6,
        ax=ax,
    )
    ax.set_title("Age vs Charges by Smoking Status")
    ax.set_xlabel("Age (Years)")
    ax.set_ylabel("Predicted Charges ($)")
    st.pyplot(fig)
    plt.close(fig)

# Scored Data Download Table
st.subheader("📋 Scored Portfolio Records")
st.dataframe(filtered.head(50), width="stretch")

csv_data = filtered.to_csv(index=False).encode("utf-8")
st.download_button(
    label="📥 Download Filtered Portfolio (CSV)",
    data=csv_data,
    file_name="claimsight_scored_portfolio.csv",
    mime="text/csv",
)
