import pandas as pd

from claimsight.config import Settings


def build_cost_features(df: pd.DataFrame, cfg: Settings) -> pd.DataFrame:
    """Build engineered feature set for medical cost estimation."""
    df_feat = df.copy()

    # Binned categorical features
    # Extend upper bound slightly to accommodate inference range without altering training bins
    age_bins = list(cfg.cost.age_bins)
    if age_bins[-1] < 120:
        age_bins[-1] = 120
    df_feat["age_band"] = pd.cut(
        df_feat["age"],
        bins=age_bins,
        labels=cfg.cost.age_labels,
        include_lowest=True,
    ).astype(str)

    bmi_bins = list(cfg.cost.bmi_bins)
    if bmi_bins[-1] < 150:
        bmi_bins[-1] = 150
    df_feat["bmi_category"] = pd.cut(
        df_feat["bmi"],
        bins=bmi_bins,
        labels=cfg.cost.bmi_labels,
        right=False,
    ).astype(str)

    # Indicator flags
    df_feat["is_obese"] = (df_feat["bmi"] >= 30.0).astype(int)
    df_feat["smoker_flag"] = (df_feat["smoker"].astype(str).str.lower() == "yes").astype(int)
    df_feat["sex_flag"] = (df_feat["sex"].astype(str).str.lower() == "male").astype(int)
    df_feat["has_children"] = (df_feat["children"] > 0).astype(int)

    # Clinical & actuarial interactions
    df_feat["smoker_x_obese"] = df_feat["smoker_flag"] * df_feat["is_obese"]
    df_feat["smoker_x_age"] = df_feat["smoker_flag"] * df_feat["age"]
    df_feat["bmi_x_smoker"] = df_feat["bmi"] * df_feat["smoker_flag"]

    # Family size grouping
    def _family_bucket(c: int) -> str:
        if c == 0:
            return "0"
        elif c in (1, 2):
            return "1-2"
        return "3+"

    df_feat["family_size_bucket"] = df_feat["children"].apply(_family_bucket).astype(str)

    # Ensure no NaN or inf in model features
    feature_cols = [
        "age",
        "bmi",
        "children",
        "sex",
        "smoker",
        "region",
        "age_band",
        "bmi_category",
        "is_obese",
        "smoker_flag",
        "sex_flag",
        "has_children",
        "smoker_x_obese",
        "smoker_x_age",
        "bmi_x_smoker",
        "family_size_bucket",
    ]
    for c in feature_cols:
        if df_feat[c].isna().any():
            raise ValueError(f"Feature column contains NaN: {c}")

    return df_feat


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate input features and medical charges target."""
    if "charges" not in df.columns:
        raise ValueError("Target column 'charges' missing from DataFrame")

    feature_cols = [
        "age",
        "bmi",
        "children",
        "sex",
        "smoker",
        "region",
        "age_band",
        "bmi_category",
        "is_obese",
        "smoker_flag",
        "sex_flag",
        "has_children",
        "smoker_x_obese",
        "smoker_x_age",
        "bmi_x_smoker",
        "family_size_bucket",
    ]
    X = df[feature_cols].copy()
    y = df["charges"].copy()
    return X, y
