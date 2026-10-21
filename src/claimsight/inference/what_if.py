from typing import Any

import pandas as pd

from claimsight.api.schemas import CostInput
from claimsight.inference.predict_cost import predict_cost


def what_if(base_inputs: dict[str, Any], changes: dict[str, Any]) -> dict[str, Any]:
    """Calculate marginal counterfactual predictions by applying attribute changes sequentially."""
    valid_fields = set(CostInput.model_fields.keys())
    unknown_keys = set(changes.keys()) - valid_fields
    if unknown_keys:
        raise ValueError(f"Unknown change parameter(s): {sorted(unknown_keys)}")

    baseline = predict_cost(base_inputs)
    new = predict_cost({**base_inputs, **changes})

    delta_abs = round(new["prediction"] - baseline["prediction"], 2)
    delta_pct = round((delta_abs / max(baseline["prediction"], 1.0)) * 100.0, 2)

    # Step-by-step sequential application
    steps = []
    current_state = dict(base_inputs)
    prev_pred = baseline["prediction"]

    for feat, new_val in changes.items():
        current_state[feat] = new_val
        step_res = predict_cost(current_state)
        step_pred = step_res["prediction"]
        step_delta = round(step_pred - prev_pred, 2)
        steps.append(
            {
                "feature": feat,
                "new_value": new_val,
                "cumulative_prediction": step_pred,
                "marginal_delta": step_delta,
            }
        )
        prev_pred = step_pred

    return {
        "baseline": baseline,
        "new": new,
        "delta_abs": delta_abs,
        "delta_pct": delta_pct,
        "steps": steps,
    }


def what_if_sweep(
    base_inputs: dict[str, Any],
    feature: str,
    values: list[Any],
) -> pd.DataFrame:
    """Generate parameter sweep curve for interactive sensitivity analysis."""
    valid_fields = set(CostInput.model_fields.keys())
    if feature not in valid_fields:
        raise ValueError(f"Unknown feature for sweep: {feature}")

    records = []
    for val in values:
        test_inputs = {**base_inputs, feature: val}
        res = predict_cost(test_inputs)
        records.append(
            {
                "feature_value": val,
                "prediction": res["prediction"],
                "lower": res["lower"],
                "upper": res["upper"],
            }
        )

    return pd.DataFrame(records)
