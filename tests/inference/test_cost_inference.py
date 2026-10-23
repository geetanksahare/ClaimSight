import pandas as pd
import pytest

from claimsight.inference.predict_cost import predict_cost
from claimsight.inference.what_if import what_if, what_if_sweep


def test_predict_cost_dict_and_dataframe_agree():
    sample = {
        "age": 30,
        "sex": "female",
        "bmi": 26.5,
        "children": 1,
        "smoker": "no",
        "region": "northwest",
    }
    res_dict = predict_cost(sample)
    df_sample = pd.DataFrame([sample])
    res_df = predict_cost(df_sample)

    assert round(res_dict["prediction"], 2) == round(res_df.loc[0, "prediction"], 2)
    assert round(res_dict["lower"], 2) == round(res_df.loc[0, "lower"], 2)
    assert round(res_dict["upper"], 2) == round(res_df.loc[0, "upper"], 2)


def test_out_of_range_flag():
    # Age 85 is valid according to CostInput schema (18 to 100), but outside training range [18, 64]
    sample_extrapolated = {
        "age": 85,
        "sex": "male",
        "bmi": 25.0,
        "children": 0,
        "smoker": "no",
        "region": "southwest",
    }
    res = predict_cost(sample_extrapolated)
    assert res["in_training_range"] is False
    assert len(res["warnings"]) > 0
    assert "outside training range" in res["warnings"][0]


def test_what_if_zero_delta_on_identical():
    sample = {
        "age": 40,
        "sex": "male",
        "bmi": 28.0,
        "children": 2,
        "smoker": "no",
        "region": "southeast",
    }
    res = what_if(sample, {})
    assert res["delta_abs"] == 0.0
    assert res["delta_pct"] == 0.0
    assert len(res["steps"]) == 0


def test_what_if_rejects_unknown_keys():
    sample = {
        "age": 40,
        "sex": "male",
        "bmi": 28.0,
        "children": 2,
        "smoker": "no",
        "region": "southeast",
    }
    with pytest.raises(ValueError, match="Unknown change parameter"):
        what_if(sample, {"income": 75000})


def test_what_if_step_consistency():
    sample = {
        "age": 35,
        "sex": "female",
        "bmi": 24.0,
        "children": 0,
        "smoker": "no",
        "region": "northeast",
    }
    changes = {"smoker": "yes", "bmi": 32.0}
    res = what_if(sample, changes)

    # Final step cumulative prediction equals the combined counterfactual prediction
    final_step_pred = res["steps"][-1]["cumulative_prediction"]
    assert final_step_pred == res["new"]["prediction"]


def test_what_if_sweep_shape():
    sample = {
        "age": 35,
        "sex": "female",
        "bmi": 24.0,
        "children": 0,
        "smoker": "no",
        "region": "northeast",
    }
    vals = [20.0, 25.0, 30.0, 35.0]
    sweep_df = what_if_sweep(sample, "bmi", vals)
    assert len(sweep_df) == 4
    assert list(sweep_df.columns) == ["feature_value", "prediction", "lower", "upper"]


def test_what_if_arrow_serialization_safety():
    """Verify that multi-step sequential counterfactuals serialize to PyArrow Table without ArrowTypeError."""
    import pandas as pd
    import pyarrow as pa

    sample = {
        "age": 35,
        "sex": "female",
        "bmi": 24.0,
        "children": 0,
        "smoker": "no",
        "region": "northeast",
    }
    changes = {"smoker": "yes", "bmi": 32.5}
    res = what_if(sample, changes)
    steps_df = pd.DataFrame(res["steps"])
    # Arrow conversion will fail if new_value has mixed types
    table = pa.Table.from_pandas(steps_df)
    assert table.num_rows == 2
    assert all(isinstance(val, str) for val in steps_df["new_value"])


def test_cost_boundary_inputs():
    """Verify inference works on boundary conditions: BMI 29.99 vs 30.0, age 18 and 64, children 0 and 5."""
    base_case = {
        "age": 18,
        "sex": "male",
        "bmi": 29.99,
        "children": 0,
        "smoker": "no",
        "region": "southwest",
    }
    res_under_30 = predict_cost(base_case)
    assert res_under_30["prediction"] > 0
    assert res_under_30["in_training_range"] is True

    # Boundary at exactly 30.0 (obesity cutoff)
    obese_case = dict(base_case, bmi=30.0, age=64, children=5)
    res_obese = predict_cost(obese_case)
    assert res_obese["prediction"] > res_under_30["prediction"]
    assert res_obese["in_training_range"] is True
