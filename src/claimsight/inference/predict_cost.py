from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from claimsight.api.schemas import CostInput
from claimsight.config import get_settings
from claimsight.features.cost_features import build_cost_features, split_xy
from claimsight.models.conformal import predict_interval
from claimsight.models.registry import load_model


@lru_cache(maxsize=1)
def _get_cost_model_bundle() -> tuple[Any, dict[str, Any]]:
    return load_model("cost")


def _check_training_range(
    row: dict[str, Any], numeric_ranges: dict[str, dict[str, float]]
) -> tuple[bool, list[str]]:
    in_range = True
    warnings_list = []

    for col in ["age", "bmi", "children"]:
        if col in row and col in numeric_ranges:
            val = float(row[col])
            c_min = float(numeric_ranges[col].get("min", 0.0))
            c_max = float(numeric_ranges[col].get("max", 100.0))
            if val < c_min or val > c_max:
                in_range = False
                warnings_list.append(
                    f"Feature '{col}' value {val} is outside training range [{c_min}, {c_max}]. "
                    "Prediction involves model extrapolation."
                )
    return in_range, warnings_list


def predict_cost(
    inputs: dict[str, Any] | pd.DataFrame,
    alpha: float | None = None,
) -> dict[str, Any] | pd.DataFrame:
    """Predict annual medical charges with calibrated conformal prediction intervals."""
    cfg = get_settings()
    model, metadata = _get_cost_model_bundle()

    conformal_q = float(metadata.get("conformal_q", 0.35))
    stored_alpha = float(metadata.get("alpha", cfg.cost.interval_alpha))
    eval_alpha = alpha if alpha is not None else stored_alpha
    coverage_level = round(1.0 - eval_alpha, 2)
    numeric_ranges = metadata.get("numeric_ranges", {})

    is_single_dict = isinstance(inputs, dict)
    if is_single_dict:
        # Pydantic validation
        validated = CostInput(**inputs)
        df_in = pd.DataFrame([validated.model_dump()])
    else:
        # Validate each row
        records = inputs.to_dict(orient="records")
        validated_records = [CostInput(**r).model_dump() for r in records]
        df_in = pd.DataFrame(validated_records)

    features_df = build_cost_features(df_in, cfg)
    X, _ = split_xy(features_df) if "charges" in features_df.columns else (features_df, None)

    preds = model.predict(X)
    lowers, uppers = predict_interval(model, X, conformal_q)

    # Range and warning checks
    in_range_flags = []
    warnings_records = []
    for r in df_in.to_dict(orient="records"):
        in_r, warns = _check_training_range(r, numeric_ranges)
        in_range_flags.append(in_r)
        warnings_records.append(warns)

    if is_single_dict:
        return {
            "prediction": round(float(preds[0]), 2),
            "lower": round(float(lowers[0]), 2),
            "upper": round(float(uppers[0]), 2),
            "coverage_level": coverage_level,
            "in_training_range": in_range_flags[0],
            "warnings": warnings_records[0],
        }

    out_df = inputs.copy()
    out_df["prediction"] = np.round(preds, 2)
    out_df["lower"] = np.round(lowers, 2)
    out_df["upper"] = np.round(uppers, 2)
    out_df["coverage_level"] = coverage_level
    out_df["in_training_range"] = in_range_flags
    out_df["warnings"] = warnings_records
    return out_df
