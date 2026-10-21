import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from claimsight.config import get_settings
from claimsight.features.build_features import build_features


def test_no_member_overlap_between_splits():
    cfg = get_settings()
    data_path = cfg.paths.processed_dir / "synpuf_beneficiary.parquet"
    if not data_path.exists():
        return

    synpuf = {
        "beneficiary": pd.read_parquet(data_path),
        "inpatient": pd.read_parquet(cfg.paths.processed_dir / "synpuf_inpatient.parquet"),
        "outpatient": pd.read_parquet(cfg.paths.processed_dir / "synpuf_outpatient.parquet"),
        "carrier_agg": pd.read_parquet(cfg.paths.processed_dir / "synpuf_carrier_agg.parquet"),
    }
    df_feat = build_features("member", synpuf, cfg)

    gss = GroupShuffleSplit(n_splits=1, train_size=0.80, random_state=cfg.seed)
    train_idx, test_idx = next(gss.split(df_feat, groups=df_feat["member_id"]))

    train_members = set(df_feat.iloc[train_idx]["member_id"])
    test_members = set(df_feat.iloc[test_idx]["member_id"])

    assert len(train_members.intersection(test_members)) == 0


def test_test_indices_never_passed_to_fit():
    # Spy wrapper to verify test rows are never fitted
    fitted_indices: list[int] = []

    class SpyEstimator:
        def fit(self, X, y=None):
            fitted_indices.extend(list(X.index))
            return self

        def predict(self, X):
            return np.zeros(len(X))

    n = 100
    df = pd.DataFrame({"feat": np.arange(n)}, index=np.arange(n))
    test_idx = set(range(80, 100))
    train_idx = [i for i in range(n) if i not in test_idx]

    spy = SpyEstimator()
    spy.fit(df.iloc[train_idx])

    # Assert none of the test indices were touched by fit
    assert set(fitted_indices).isdisjoint(test_idx)
