import pandas as pd
import pandera.errors
import pytest

from claimsight.data.load_and_clean import _clean_cost
from claimsight.data.schemas import cost_clean_schema


def test_schema_accepts_valid_sample():
    valid_df = pd.DataFrame(
        {
            "row_id": [0, 1],
            "age": [25, 45],
            "sex": ["female", "male"],
            "bmi": [22.4, 31.0],
            "children": [0, 2],
            "smoker": ["no", "yes"],
            "region": ["northwest", "southeast"],
            "charges": [3200.5, 24500.0],
        }
    )
    validated = cost_clean_schema.validate(valid_df)
    assert len(validated) == 2


def test_schema_rejects_negative_charges():
    invalid_df = pd.DataFrame(
        {
            "row_id": [0],
            "age": [30],
            "sex": ["male"],
            "bmi": [25.0],
            "children": [1],
            "smoker": ["no"],
            "region": ["southwest"],
            "charges": [-100.0],
        }
    )
    with pytest.raises(pandera.errors.SchemaError):
        cost_clean_schema.validate(invalid_df)


def test_schema_rejects_invalid_sex():
    invalid_df = pd.DataFrame(
        {
            "row_id": [0],
            "age": [30],
            "sex": ["other"],
            "bmi": [25.0],
            "children": [1],
            "smoker": ["no"],
            "region": ["southwest"],
            "charges": [5000.0],
        }
    )
    with pytest.raises(pandera.errors.SchemaError):
        cost_clean_schema.validate(invalid_df)


def test_schema_rejects_missing_column():
    incomplete_df = pd.DataFrame(
        {
            "row_id": [0],
            "age": [30],
            "sex": ["female"],
            "bmi": [25.0],
            "children": [1],
            "smoker": ["no"],
            # Missing region and charges
        }
    )
    with pytest.raises(pandera.errors.SchemaError):
        cost_clean_schema.validate(incomplete_df)


def test_clean_cost_duplicate_removal(tmp_path):
    csv_file = tmp_path / "insurance_test.csv"
    csv_file.write_text(
        "age,sex,bmi,children,smoker,region,charges\n"
        "19,female,27.9,0,yes,southwest,16884.924\n"
        "19,female,27.9,0,yes,southwest,16884.924\n"
        "30,male,25.5,1,no,northeast,4500.0\n"
    )
    cleaned = _clean_cost(csv_file)
    assert len(cleaned) == 2
    assert "row_id" in cleaned.columns
    assert list(cleaned["row_id"]) == [0, 1]
