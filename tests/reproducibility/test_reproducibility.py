import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from claimsight.config import get_settings
from claimsight.features.build_features import build_features
from claimsight.features.cost_features import split_xy
from claimsight.models.zoo import get_model_zoo


def test_cost_training_reproducibility():
    cfg = get_settings()
    data_path = cfg.paths.processed_dir / "cost.parquet"
    if not data_path.exists():
        from claimsight.data.load_and_clean import load_and_clean

        df = load_and_clean("cost", save=True)
    else:
        df = pd.read_parquet(data_path)

    df_feat = build_features("cost", df, cfg)

    # Run 1
    df_tr_cal1, df_te1 = train_test_split(
        df_feat, test_size=0.15, stratify=df_feat["smoker"], random_state=42
    )
    df_tr1, _ = train_test_split(
        df_tr_cal1, test_size=0.15 / 0.85, stratify=df_tr_cal1["smoker"], random_state=42
    )
    X_tr1, y_tr1 = split_xy(df_tr1)
    X_te1, y_te1 = split_xy(df_te1)

    model1 = get_model_zoo("cost", seed=42)["xgboost"]
    model1.fit(X_tr1, y_tr1)
    preds1 = model1.predict(X_te1)
    rmse1 = float(np.sqrt(np.mean((y_te1 - preds1) ** 2)))

    # Run 2
    df_tr_cal2, df_te2 = train_test_split(
        df_feat, test_size=0.15, stratify=df_feat["smoker"], random_state=42
    )
    df_tr2, _ = train_test_split(
        df_tr_cal2, test_size=0.15 / 0.85, stratify=df_tr_cal2["smoker"], random_state=42
    )
    X_tr2, y_tr2 = split_xy(df_tr2)
    X_te2, y_te2 = split_xy(df_te2)

    model2 = get_model_zoo("cost", seed=42)["xgboost"]
    model2.fit(X_tr2, y_tr2)
    preds2 = model2.predict(X_te2)
    rmse2 = float(np.sqrt(np.mean((y_te2 - preds2) ** 2)))

    # Exactly identical RMSE across runs
    assert round(rmse1, 6) == round(rmse2, 6)
    np.testing.assert_allclose(preds1, preds2, rtol=1e-5)
