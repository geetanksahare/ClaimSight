import pandas as pd

from claimsight.config import get_settings
from claimsight.features.member_features import build_member_features, label_high_cost


def test_member_features_no_target_year_leakage():
    cfg = get_settings()

    # Synthetic multi-year member table
    bene = pd.DataFrame(
        {
            "DESYNPUF_ID": ["MEM_01", "MEM_01", "MEM_01"],
            "year": [2008, 2009, 2010],
            "BENE_BIRTH_DT": pd.to_datetime(["1945-01-01"] * 3),
            "BENE_DEATH_DT": [pd.NaT] * 3,
            "BENE_SEX_IDENT_CD": ["female"] * 3,
            "BENE_RACE_CD": ["1"] * 3,
            "BENE_ESRD_IND": [0] * 3,
            "SP_STATE_CODE": ["01"] * 3,
            "BENE_HI_CVRAGE_TOT_MONS": [12] * 3,
            "BENE_SMI_CVRAGE_TOT_MONS": [12] * 3,
            "SP_ALZHDMTA": [0, 0, 1],  # Becomes 1 in 2010 (target year for 2009 base)
            "SP_CHF": [0, 0, 0],
            "MEDREIMB_IP": [0.0, 500.0, 2000.0],
            "MEDREIMB_OP": [100.0, 200.0, 800.0],
            "MEDREIMB_CAR": [50.0, 100.0, 400.0],
        }
    )

    data = {"beneficiary": bene}
    feat_df = build_member_features(data, cfg)

    # Base year 2008 features must reflect 2008 conditions (ALZHDMTA = 0), not 2010
    row_2008 = feat_df[feat_df["base_year"] == 2008].iloc[0]
    assert row_2008["prior_cost_ip"] == 0.0
    assert row_2008["SP_ALZHDMTA"] == 0
    # Next year cost is 2009 cost (500 + 200 + 100 = 800)
    assert row_2008["next_year_cost"] == 800.0

    # Base year 2009 features must reflect 2009 conditions (ALZHDMTA = 0, even though 1 in 2010)
    row_2009 = feat_df[feat_df["base_year"] == 2009].iloc[0]
    assert row_2009["SP_ALZHDMTA"] == 0
    # Next year cost is 2010 cost (2000 + 800 + 400 = 3200)
    assert row_2009["next_year_cost"] == 3200.0


def test_high_cost_threshold_calculated_on_train_only():
    y_train = pd.Series(
        [1000.0, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0, 7000.0, 8000.0, 9000.0, 10000.0]
    )
    y_test = pd.Series([500.0, 50000.0])  # extreme outlier in test
    y_all = pd.concat([y_train, y_test], ignore_index=True)

    # 90th percentile of y_train only
    q = 0.90
    threshold, labels = label_high_cost(y_train, y_all, q=q)

    expected_thresh = float(y_train.quantile(q))
    assert threshold == expected_thresh

    # Test outlier (50,000) is flagged as high cost without inflating training threshold
    assert labels.iloc[-1] == 1
    assert labels.iloc[-2] == 0
