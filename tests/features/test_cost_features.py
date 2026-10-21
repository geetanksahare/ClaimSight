import numpy as np
import pandas as pd

from claimsight.config import get_settings
from claimsight.features.cost_features import build_cost_features, split_xy


def test_bmi_category_and_obesity_boundary():
    cfg = get_settings()
    sample = pd.DataFrame(
        {
            "age": [25, 35],
            "sex": ["male", "female"],
            "bmi": [29.9, 30.0],
            "children": [0, 2],
            "smoker": ["no", "yes"],
            "region": ["northwest", "southeast"],
            "charges": [3000.0, 25000.0],
        }
    )
    feat = build_cost_features(sample, cfg)

    # 29.9 is overweight, not obese
    assert feat.loc[0, "bmi_category"] == "overweight"
    assert feat.loc[0, "is_obese"] == 0

    # 30.0 is obese
    assert feat.loc[1, "bmi_category"] == "obese"
    assert feat.loc[1, "is_obese"] == 1


def test_cost_features_no_nan_or_inf():
    cfg = get_settings()
    sample = pd.DataFrame(
        {
            "age": [19, 45, 60],
            "sex": ["female", "male", "female"],
            "bmi": [18.0, 27.5, 34.2],
            "children": [0, 1, 4],
            "smoker": ["yes", "no", "no"],
            "region": ["southwest", "northeast", "southeast"],
            "charges": [15000.0, 7500.0, 12000.0],
        }
    )
    feat = build_cost_features(sample, cfg)
    X, y = split_xy(feat)

    assert not X.isna().any().any()
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    assert not np.isinf(X[numeric_cols].to_numpy()).any()


def test_family_size_buckets_and_interactions():
    cfg = get_settings()
    sample = pd.DataFrame(
        {
            "age": [30, 40, 50],
            "sex": ["female", "male", "female"],
            "bmi": [32.0, 24.0, 31.0],
            "children": [0, 2, 4],
            "smoker": ["yes", "no", "yes"],
            "region": ["northwest", "southwest", "northeast"],
            "charges": [20000.0, 6000.0, 28000.0],
        }
    )
    feat = build_cost_features(sample, cfg)

    assert list(feat["family_size_bucket"]) == ["0", "1-2", "3+"]
    assert feat.loc[0, "smoker_x_obese"] == 1
    assert feat.loc[1, "smoker_x_obese"] == 0
    assert feat.loc[0, "smoker_x_age"] == 30
    assert feat.loc[0, "bmi_x_smoker"] == 32.0
