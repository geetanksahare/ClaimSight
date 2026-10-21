import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

from claimsight.config import get_settings
from claimsight.models.conformal import evaluate_conformal_coverage, fit_conformal, predict_interval
from claimsight.models.registry import load_model, save_model
from claimsight.models.train_and_compare import get_default_regression_scorers, train_and_compare
from claimsight.models.tune import tune_model
from claimsight.models.zoo import get_model_zoo


def test_leaderboard_sorted_and_complete():
    cfg = get_settings()
    # Synthetic small dataset
    rng = np.random.RandomState(cfg.seed)
    n = 100
    df = pd.DataFrame(
        {
            "age": rng.randint(18, 64, size=n),
            "bmi": rng.uniform(18.0, 45.0, size=n),
            "children": rng.randint(0, 5, size=n),
            "sex": rng.choice(["male", "female"], size=n),
            "smoker": rng.choice(["yes", "no"], size=n),
            "region": rng.choice(["northeast", "northwest", "southeast", "southwest"], size=n),
            "age_band": "18-29",
            "bmi_category": "normal",
            "is_obese": 0,
            "smoker_flag": 0,
            "sex_flag": 1,
            "has_children": 0,
            "smoker_x_obese": 0,
            "smoker_x_age": 0,
            "bmi_x_smoker": 0.0,
            "family_size_bucket": "0",
        }
    )
    y = pd.Series(rng.uniform(2000, 30000, size=n))

    zoo = get_model_zoo("cost", seed=cfg.seed)
    cv = KFold(n_splits=3, shuffle=True, random_state=cfg.seed)
    scorers = get_default_regression_scorers()

    lb = train_and_compare(df, y, zoo=zoo, cv=cv, scoring=scorers, primary_metric="rmse")

    assert len(lb) == len(zoo)
    assert set(lb["model"]) == set(zoo.keys())
    # Best RMSE first (ascending sort)
    assert lb["rmse_mean"].is_monotonic_increasing


def test_tune_model_within_bounds():
    cfg = get_settings()
    rng = np.random.RandomState(cfg.seed)
    n = 60
    df = pd.DataFrame(
        {
            "age": rng.randint(18, 64, size=n),
            "bmi": rng.uniform(18.0, 45.0, size=n),
            "children": rng.randint(0, 5, size=n),
            "sex": "male",
            "smoker": "no",
            "region": "southwest",
            "age_band": "30-39",
            "bmi_category": "normal",
            "is_obese": 0,
            "smoker_flag": 0,
            "sex_flag": 1,
            "has_children": 0,
            "smoker_x_obese": 0,
            "smoker_x_age": 0,
            "bmi_x_smoker": 0.0,
            "family_size_bucket": "0",
        }
    )
    y = pd.Series(rng.uniform(2000, 30000, size=n))

    zoo = get_model_zoo("cost", seed=cfg.seed)
    base = zoo["lightgbm"]
    cv = KFold(n_splits=3, shuffle=True, random_state=cfg.seed)

    best_est, best_params = tune_model(
        "test_lgb", base, df, y, cv=cv, scoring_metric="rmse", n_trials=3, seed=cfg.seed
    )

    assert "n_estimators" in best_params
    assert 200 <= best_params["n_estimators"] <= 1200
    assert 0.01 <= best_params["learning_rate"] <= 0.2


def test_conformal_coverage_synthetic():
    rng = np.random.RandomState(42)
    n_cal = 1500
    n_test = 1500

    # True model with known log-normal noise
    class MockModel:
        def predict(self, X):
            return np.ones(len(X)) * 10000.0

    model = MockModel()
    X_cal = pd.DataFrame({"a": np.zeros(n_cal)})
    X_test = pd.DataFrame({"a": np.zeros(n_test)})

    # Simulated targets with log standard deviation sigma = 0.2
    y_cal = 10000.0 * np.exp(rng.normal(0, 0.2, size=n_cal))
    y_test = 10000.0 * np.exp(rng.normal(0, 0.2, size=n_test))

    alpha = 0.10
    q = fit_conformal(model, X_cal, y_cal, alpha=alpha)
    lower, upper = predict_interval(model, X_test, q)
    cov = evaluate_conformal_coverage(y_test, lower, upper)

    # Within 3 points (0.03) of nominal coverage 1 - alpha = 0.90
    assert abs(cov - 0.90) <= 0.03


def test_registry_save_load_round_trip(tmp_path, monkeypatch):
    # Model and predictions
    model = Ridge().fit([[1, 2], [3, 4]], [10.0, 20.0])
    preds_orig = model.predict([[2, 3]])

    meta = {
        "version": "v_test",
        "training_rows": 2,
        "features": ["f1", "f2"],
        "numeric_ranges": {"f1": {"min": 1, "max": 3}},
    }
    save_model("cost_test", model, meta)
    loaded_model, loaded_meta = load_model("cost_test", version="v_test")

    preds_loaded = loaded_model.predict([[2, 3]])
    np.testing.assert_allclose(preds_orig, preds_loaded)
    assert loaded_meta["training_rows"] == 2


def test_registry_sklearn_version_warning():
    model = Ridge().fit([[1], [2]], [1.0, 2.0])
    meta = {
        "version": "v_mismatch",
        "sklearn_version": "0.19.1",
    }
    save_model("cost_test_mismatch", model, meta)

    with pytest.warns(UserWarning, match="was trained with scikit-learn 0.19.1"):
        load_model("cost_test_mismatch", version="v_mismatch")
