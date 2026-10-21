import numpy as np
import pandas as pd
import shap

from claimsight.explain.explain_prediction import _extract_components, explain_prediction
from claimsight.models.registry import load_model


def test_shap_reconstruction_log_scale():
    sample = {
        "age": 35,
        "sex": "female",
        "bmi": 26.0,
        "children": 1,
        "smoker": "no",
        "region": "northwest",
    }
    model, metadata = load_model("cost")
    res = explain_prediction("cost", sample, top_k=20, model_bundle=(model, metadata))

    # Base value + sum of all local SHAP values equals log1p(prediction)
    base_val = res["base_value"]
    total_shap = sum(c["shap"] for c in res["contributions"])
    reconstructed_log = base_val + total_shap

    pred_dollar = res["prediction"]
    actual_log = np.log1p(pred_dollar)

    # Within tolerance on log scale
    np.testing.assert_allclose(reconstructed_log, actual_log, atol=1e-2)


def test_one_hot_aggregation_preserves_total():
    model, metadata = load_model("cost")
    preprocessor, inner_model = _extract_components(model)

    sample = {
        "age": 42,
        "sex": "male",
        "bmi": 31.5,
        "children": 2,
        "smoker": "yes",
        "region": "southeast",
    }
    res = explain_prediction("cost", sample, top_k=20, model_bundle=(model, metadata))

    # Also compute raw unaggregated SHAP on transformed vector
    from claimsight.config import get_settings
    from claimsight.features.build_features import build_features

    cfg = get_settings()
    feat = build_features("cost", pd.DataFrame([sample]), cfg)
    X = feat.copy()
    X_trans = preprocessor.transform(X)

    explainer = shap.TreeExplainer(inner_model)
    raw_shap = explainer.shap_values(X_trans)[0]

    raw_sum = float(np.sum(raw_shap))
    aggregated_sum = sum(c["shap"] for c in res["contributions"])

    # Sum of aggregated feature contributions equals sum of one-hot transformed SHAP (within 4-decimal rounding)
    np.testing.assert_allclose(raw_sum, aggregated_sum, atol=1e-3)
