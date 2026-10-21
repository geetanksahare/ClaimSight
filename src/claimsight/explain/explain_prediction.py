from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.compose import TransformedTargetRegressor

from claimsight.config import get_settings
from claimsight.explain.narrative import to_text
from claimsight.features.build_features import build_features
from claimsight.features.cost_features import split_xy as split_cost_xy
from claimsight.logging_config import get_logger
from claimsight.models.registry import load_model

logger = get_logger(__name__)


@lru_cache(maxsize=3)
def _get_cached_model_bundle(module: str) -> tuple[Any, dict[str, Any]]:
    return load_model(module)


def _extract_components(model: Any) -> tuple[Any, Any]:
    """Extract fitted ColumnTransformer preprocessor and inner tree estimator."""
    if isinstance(model, dict):
        model = model.get("classifier", model.get("regressor"))

    if isinstance(model, TransformedTargetRegressor):
        inner_pipe = model.regressor_
        preprocessor = inner_pipe.named_steps["preprocessor"]
        estimator = inner_pipe.named_steps["model"]
        return preprocessor, estimator

    if hasattr(model, "calibrated_classifiers_"):
        first_calib = model.calibrated_classifiers_[0]
        base_est = getattr(first_calib, "estimator", getattr(first_calib, "base_estimator", None))
        if hasattr(base_est, "named_steps"):
            return base_est.named_steps["preprocessor"], base_est.named_steps["model"]
        return None, base_est

    if hasattr(model, "named_steps"):
        return model.named_steps["preprocessor"], model.named_steps["model"]

    return None, model


def explain_prediction(
    module: Literal["cost", "member", "fraud"],
    inputs: dict[str, Any] | pd.DataFrame,
    top_k: int = 5,
    model_bundle: tuple[Any, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Compute local SHAP contributions and plain-language narrative for inputs."""
    cfg = get_settings()
    if model_bundle is None:
        model, metadata = _get_cached_model_bundle(module)
    else:
        model, metadata = model_bundle

    if isinstance(inputs, dict):
        input_df = pd.DataFrame([inputs])
    else:
        input_df = inputs.copy()

    if module == "cost":
        features_df = build_features(module, input_df, cfg)
        X, _ = (
            split_cost_xy(features_df) if "charges" in features_df.columns else (features_df, None)
        )
    elif module == "member":
        if "age_at_year_end" in input_df.columns:
            drop_cols = [
                c for c in ["member_id", "base_year", "next_year_cost"] if c in input_df.columns
            ]
            X = input_df.drop(columns=drop_cols)
        else:
            X = build_features(module, input_df, cfg)
    elif "is_fraud" in input_df.columns:
        X = input_df.drop(columns=["is_fraud"])
    else:
        X = input_df

    preprocessor, inner_model = _extract_components(model)
    if preprocessor is not None:
        X_transformed = preprocessor.transform(X)
        feat_names = list(preprocessor.get_feature_names_out())
    else:
        X_transformed = X.to_numpy()
        feat_names = list(X.columns)

    # Initialize SHAP explainer (TreeExplainer for tree models, LinearExplainer for linear models)
    try:
        explainer = shap.TreeExplainer(inner_model)
        shap_out = explainer.shap_values(X_transformed)
        expected_val = explainer.expected_value
    except Exception:
        explainer = shap.LinearExplainer(inner_model, X_transformed)
        shap_out = explainer.shap_values(X_transformed)
        expected_val = explainer.expected_value

    if isinstance(shap_out, list):
        shap_row = shap_out[1][0] if len(shap_out) > 1 else shap_out[0][0]
    elif hasattr(shap_out, "ndim") and shap_out.ndim == 3:
        shap_row = shap_out[0, :, 1]
    elif hasattr(shap_out, "ndim") and shap_out.ndim == 2:
        shap_row = shap_out[0]
    else:
        shap_row = np.asarray(shap_out).flatten()

    expected_val = explainer.expected_value
    if isinstance(expected_val, (list, np.ndarray)):
        expected_val = float(expected_val[-1])
    else:
        expected_val = float(expected_val)

    # Aggregate one-hot encoded columns back to source features
    grouped_shap: dict[str, float] = {}
    for col_name, s_val in zip(feat_names, shap_row):
        clean_name = col_name.split("__")[-1]
        matched_feature = None
        for orig_f in X.columns:
            if clean_name == orig_f or clean_name.startswith(f"{orig_f}_"):
                matched_feature = orig_f
                break
        if matched_feature is None:
            matched_feature = clean_name
        grouped_shap[matched_feature] = grouped_shap.get(matched_feature, 0.0) + float(s_val)

    pred_val = float(model.predict(X)[0]) if hasattr(model, "predict") else expected_val

    contributions = []
    for f_name, s_val in grouped_shap.items():
        val = X[f_name].iloc[0] if f_name in X.columns else None
        if hasattr(val, "item"):
            val = val.item()
        direction = "increases" if s_val >= 0 else "decreases"
        effect_pct = float((np.exp(s_val) - 1.0) * 100.0) if module == "cost" else None
        narrative_sentence = to_text(f_name, val, s_val, effect_pct=effect_pct, module=module)

        contributions.append(
            {
                "feature": f_name,
                "value": val,
                "shap": round(s_val, 4),
                "effect_pct": round(effect_pct, 2) if effect_pct is not None else None,
                "direction": direction,
                "text": narrative_sentence,
            }
        )

    # Sort by absolute SHAP descending and truncate to top_k
    contributions.sort(key=lambda item: abs(item["shap"]), reverse=True)
    top_contributions = contributions[:top_k]

    summary_text = (
        f"Prediction of ${pred_val:,.2f} driven primarily by "
        f"{', '.join([c['feature'] for c in top_contributions[:3]])}."
    )

    return {
        "module": module,
        "base_value": round(expected_val, 4),
        "prediction": round(pred_val, 2),
        "contributions": top_contributions,
        "summary_text": summary_text,
    }


def explain_global(module: Literal["cost", "member", "fraud"]) -> Path:
    """Generate global SHAP summary plot on background sample."""
    cfg = get_settings()
    model, metadata = _get_cached_model_bundle(module)
    preprocessor, inner_model = _extract_components(model)

    if module == "cost":
        data_path = cfg.paths.processed_dir / "cost.parquet"
        df = pd.read_parquet(data_path).sample(n=min(1000, 1337), random_state=cfg.seed)
        features_df = build_features("cost", df, cfg)
        X, _ = split_cost_xy(features_df)
    else:
        raise NotImplementedError("Global explainability implemented for cost module in Phase 2")

    if preprocessor is not None:
        X_transformed = preprocessor.transform(X)
        feat_names = list(preprocessor.get_feature_names_out())
    else:
        X_transformed = X.to_numpy()
        feat_names = list(X.columns)

    explainer = shap.TreeExplainer(inner_model)
    shap_vals = explainer.shap_values(X_transformed)

    out_path = cfg.paths.reports_dir / "figures" / f"shap_summary_{module}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(9, 6))
    shap.summary_plot(shap_vals, X_transformed, feature_names=feat_names, show=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

    return out_path
