from typing import Any, Literal

import lightgbm as lgb
import numpy as np
import xgboost as xgb
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _create_preprocessor(
    categorical_cols: list[str],
    numeric_cols: list[str],
    scale_numeric: bool = False,
) -> ColumnTransformer:
    transformers = [
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            categorical_cols,
        ),
        (
            "num",
            StandardScaler() if scale_numeric else "passthrough",
            numeric_cols,
        ),
    ]
    return ColumnTransformer(
        transformers=transformers, remainder="drop", verbose_feature_names_out=False
    )


def get_cost_columns() -> tuple[list[str], list[str]]:
    categorical = ["sex", "smoker", "region", "age_band", "bmi_category", "family_size_bucket"]
    numeric = [
        "age",
        "bmi",
        "children",
        "is_obese",
        "smoker_flag",
        "sex_flag",
        "has_children",
        "smoker_x_obese",
        "smoker_x_age",
        "bmi_x_smoker",
    ]
    return categorical, numeric


def get_model_zoo(
    task: Literal["cost", "member_reg", "member_clf", "fraud"],
    seed: int,
    **kwargs: Any,
) -> dict[str, Any]:
    """Return dictionary of configured baseline and candidate estimators."""
    zoo: dict[str, Any] = {}

    if task == "cost":
        cat_cols, num_cols = get_cost_columns()

        # 1. Dummy Baseline
        dummy_pipe = Pipeline(
            [
                ("preprocessor", _create_preprocessor(cat_cols, num_cols, scale_numeric=False)),
                ("model", DummyRegressor(strategy="mean")),
            ]
        )
        zoo["dummy"] = TransformedTargetRegressor(
            regressor=dummy_pipe, func=np.log1p, inverse_func=np.expm1
        )

        # 2. Linear Regression
        lr_pipe = Pipeline(
            [
                ("preprocessor", _create_preprocessor(cat_cols, num_cols, scale_numeric=True)),
                ("model", LinearRegression()),
            ]
        )
        zoo["linear_regression"] = TransformedTargetRegressor(
            regressor=lr_pipe, func=np.log1p, inverse_func=np.expm1
        )

        # 3. Ridge Regression
        ridge_pipe = Pipeline(
            [
                ("preprocessor", _create_preprocessor(cat_cols, num_cols, scale_numeric=True)),
                ("model", Ridge(alpha=1.0, random_state=seed)),
            ]
        )
        zoo["ridge"] = TransformedTargetRegressor(
            regressor=ridge_pipe, func=np.log1p, inverse_func=np.expm1
        )

        # 4. Random Forest Regressor
        rf_pipe = Pipeline(
            [
                ("preprocessor", _create_preprocessor(cat_cols, num_cols, scale_numeric=False)),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=400, min_samples_leaf=3, random_state=seed, n_jobs=-1
                    ),
                ),
            ]
        )
        zoo["random_forest"] = TransformedTargetRegressor(
            regressor=rf_pipe, func=np.log1p, inverse_func=np.expm1
        )

        # 5. XGBoost Regressor
        xgb_reg = xgb.XGBRegressor(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            n_jobs=-1,
        )
        xgb_pipe = Pipeline(
            [
                ("preprocessor", _create_preprocessor(cat_cols, num_cols, scale_numeric=False)),
                ("model", xgb_reg),
            ]
        )
        zoo["xgboost"] = TransformedTargetRegressor(
            regressor=xgb_pipe, func=np.log1p, inverse_func=np.expm1
        )

        # 6. LightGBM Regressor
        lgb_reg = lgb.LGBMRegressor(
            n_estimators=500,
            learning_rate=0.03,
            num_leaves=15,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            verbose=-1,
            n_jobs=-1,
        )
        lgb_pipe = Pipeline(
            [
                ("preprocessor", _create_preprocessor(cat_cols, num_cols, scale_numeric=False)),
                ("model", lgb_reg),
            ]
        )
        zoo["lightgbm"] = TransformedTargetRegressor(
            regressor=lgb_pipe, func=np.log1p, inverse_func=np.expm1
        )
        return zoo

    if task == "member_reg":
        tweedie_power = kwargs.get("tweedie_power", 1.5)
        cat_cols = kwargs.get("cat_cols", ["BENE_SEX_IDENT_CD", "BENE_RACE_CD", "SP_STATE_CODE"])
        num_cols = kwargs.get("num_cols", [])

        # Dummy
        zoo["dummy"] = Pipeline(
            [
                ("preprocessor", _create_preprocessor(cat_cols, num_cols, scale_numeric=False)),
                ("model", DummyRegressor(strategy="mean")),
            ]
        )

        # Ridge on log1p
        ridge_pipe = Pipeline(
            [
                ("preprocessor", _create_preprocessor(cat_cols, num_cols, scale_numeric=True)),
                ("model", Ridge(alpha=1.0, random_state=seed)),
            ]
        )
        zoo["ridge"] = TransformedTargetRegressor(
            regressor=ridge_pipe, func=np.log1p, inverse_func=np.expm1
        )

        # LightGBM Tweedie (no log transform)
        lgb_tweedie = lgb.LGBMRegressor(
            objective="tweedie",
            tweedie_variance_power=tweedie_power,
            n_estimators=400,
            learning_rate=0.03,
            random_state=seed,
            verbose=-1,
            n_jobs=-1,
        )
        zoo["lightgbm_tweedie"] = Pipeline(
            [
                ("preprocessor", _create_preprocessor(cat_cols, num_cols, scale_numeric=False)),
                ("model", lgb_tweedie),
            ]
        )

        # XGBoost Tweedie
        xgb_tweedie = xgb.XGBRegressor(
            objective="reg:tweedie",
            tweedie_variance_power=tweedie_power,
            n_estimators=400,
            learning_rate=0.03,
            random_state=seed,
            n_jobs=-1,
        )
        zoo["xgboost_tweedie"] = Pipeline(
            [
                ("preprocessor", _create_preprocessor(cat_cols, num_cols, scale_numeric=False)),
                ("model", xgb_tweedie),
            ]
        )
        return zoo

    if task == "member_clf":
        cat_cols = kwargs.get("cat_cols", ["BENE_SEX_IDENT_CD", "BENE_RACE_CD", "SP_STATE_CODE"])
        num_cols = kwargs.get("num_cols", [])

        zoo["dummy"] = Pipeline(
            [
                ("preprocessor", _create_preprocessor(cat_cols, num_cols, scale_numeric=False)),
                ("model", DummyClassifier(strategy="prior")),
            ]
        )

        lr = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
        zoo["logistic_regression"] = Pipeline(
            [
                ("preprocessor", _create_preprocessor(cat_cols, num_cols, scale_numeric=True)),
                ("model", lr),
            ]
        )

        lgb_clf = lgb.LGBMClassifier(random_state=seed, verbose=-1, n_jobs=-1)
        zoo["lightgbm"] = Pipeline(
            [
                ("preprocessor", _create_preprocessor(cat_cols, num_cols, scale_numeric=False)),
                ("model", lgb_clf),
            ]
        )

        xgb_clf = xgb.XGBClassifier(random_state=seed, n_jobs=-1)
        zoo["xgboost"] = Pipeline(
            [
                ("preprocessor", _create_preprocessor(cat_cols, num_cols, scale_numeric=False)),
                ("model", xgb_clf),
            ]
        )
        return zoo

    if task == "fraud":
        scale_pos_weight = kwargs.get("scale_pos_weight", 1.0)
        num_cols = kwargs.get("num_cols", [])
        cat_cols = kwargs.get("cat_cols", [])

        zoo["dummy"] = Pipeline(
            [
                ("preprocessor", _create_preprocessor(cat_cols, num_cols, scale_numeric=False)),
                ("model", DummyClassifier(strategy="prior")),
            ]
        )

        zoo["logistic_regression"] = Pipeline(
            [
                ("preprocessor", _create_preprocessor(cat_cols, num_cols, scale_numeric=True)),
                (
                    "model",
                    LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed),
                ),
            ]
        )

        zoo["random_forest"] = Pipeline(
            [
                ("preprocessor", _create_preprocessor(cat_cols, num_cols, scale_numeric=False)),
                (
                    "model",
                    RandomForestClassifier(
                        class_weight="balanced_subsample",
                        n_estimators=300,
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

        zoo["xgboost"] = Pipeline(
            [
                ("preprocessor", _create_preprocessor(cat_cols, num_cols, scale_numeric=False)),
                (
                    "model",
                    xgb.XGBClassifier(
                        scale_pos_weight=scale_pos_weight,
                        n_estimators=300,
                        learning_rate=0.03,
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

        zoo["lightgbm"] = Pipeline(
            [
                ("preprocessor", _create_preprocessor(cat_cols, num_cols, scale_numeric=False)),
                (
                    "model",
                    lgb.LGBMClassifier(
                        is_unbalance=True,
                        n_estimators=300,
                        learning_rate=0.03,
                        random_state=seed,
                        verbose=-1,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        return zoo

    raise ValueError(f"Unknown task type: {task}")
