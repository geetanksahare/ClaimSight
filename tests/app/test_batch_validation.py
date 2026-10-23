import io

import pandas as pd
import pytest

from claimsight.inference.predict_cost import predict_cost


def validate_batch_cost_df(df: pd.DataFrame, max_rows: int = 50000) -> list[dict]:
    """Mirror schema validation rules enforced in Batch Scoring page."""
    if df.empty:
        return [{"row": "All", "column": "All", "issue": "CSV is empty"}]

    if len(df) > max_rows:
        return [{"row": "All", "column": "All", "issue": f"Exceeds max rows {max_rows}"}]

    errors = []
    required_cols = ["age", "sex", "bmi", "children", "smoker", "region"]
    for col in required_cols:
        if col not in df.columns:
            errors.append(
                {"row": "All", "column": col, "issue": f"Missing required column: '{col}'"}
            )

    if errors:
        return errors

    for idx, row in df.iterrows():
        try:
            age = int(row["age"])
            if not (18 <= age <= 100):
                errors.append(
                    {"row": idx + 1, "column": "age", "issue": f"Age {age} out of range [18, 100]"}
                )
        except Exception:
            errors.append({"row": idx + 1, "column": "age", "issue": "Age must be numeric integer"})

        if str(row["sex"]).lower() not in ["male", "female"]:
            errors.append({"row": idx + 1, "column": "sex", "issue": f"Invalid sex '{row['sex']}'"})

        try:
            bmi = float(row["bmi"])
            if not (10.0 <= bmi <= 70.0):
                errors.append(
                    {"row": idx + 1, "column": "bmi", "issue": f"BMI {bmi} out of range [10, 70]"}
                )
        except Exception:
            errors.append({"row": idx + 1, "column": "bmi", "issue": "BMI must be numeric float"})

        if str(row["smoker"]).lower() not in ["yes", "no"]:
            errors.append(
                {
                    "row": idx + 1,
                    "column": "smoker",
                    "issue": f"Invalid smoker flag '{row['smoker']}'",
                }
            )

    return errors


def test_empty_csv_upload_detection():
    # Case 1: Header-only CSV with zero data rows
    header_only_csv = "age,sex,bmi,children,smoker,region\n"
    df = pd.read_csv(io.StringIO(header_only_csv))
    assert df.empty is True
    errs = validate_batch_cost_df(df)
    assert len(errs) > 0
    assert "empty" in errs[0]["issue"].lower()

    # Case 2: Zero-byte upload triggers EmptyDataError in pd.read_csv
    with pytest.raises(pd.errors.EmptyDataError):
        pd.read_csv(io.StringIO(""))


def test_csv_wrong_columns_detection():
    bad_csv = "patient_id,weight,height,smoker\n1,70,175,no\n"
    df = pd.read_csv(io.StringIO(bad_csv))
    errs = validate_batch_cost_df(df)
    missing_cols = [e["column"] for e in errs]
    assert "age" in missing_cols
    assert "bmi" in missing_cols
    assert "sex" in missing_cols


def test_csv_over_row_limit():
    rows = [
        {
            "age": 30,
            "sex": "male",
            "bmi": 25.0,
            "children": 0,
            "smoker": "no",
            "region": "northeast",
        }
    ] * 105
    df = pd.DataFrame(rows)
    errs = validate_batch_cost_df(df, max_rows=100)
    assert len(errs) == 1
    assert "Exceeds max rows" in errs[0]["issue"]


def test_extreme_valid_inputs_inference():
    """Verify inference gracefully handles extreme but valid inputs."""
    extreme_inputs = pd.DataFrame(
        [
            # Youngest, lowest valid BMI, 0 children
            {
                "age": 18,
                "sex": "female",
                "bmi": 10.0,
                "children": 0,
                "smoker": "no",
                "region": "southwest",
            },
            # Oldest, boundary BMI 70.0, max 10 children, heavy smoker
            {
                "age": 100,
                "sex": "male",
                "bmi": 70.0,
                "children": 10,
                "smoker": "yes",
                "region": "southeast",
            },
        ]
    )
    scored = predict_cost(extreme_inputs)
    assert len(scored) == 2
    assert (scored["prediction"] > 0).all()
    assert (scored["upper"] >= scored["lower"]).all()
