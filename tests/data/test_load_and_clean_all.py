import pytest

from claimsight.data.load_and_clean import load_and_clean


def test_load_and_clean_all_modules():
    # Cost
    cost_data = load_and_clean("cost")
    assert not cost_data.empty
    assert "charges" in cost_data.columns

    # SynPUF
    synpuf_dict = load_and_clean("synpuf")
    assert "beneficiary" in synpuf_dict
    assert not synpuf_dict["beneficiary"].empty

    # Fraud
    fraud_dict = load_and_clean("fraud")
    assert "inpatient" in fraud_dict
    assert "outpatient" in fraud_dict
    assert "beneficiary" in fraud_dict
    assert "labels" in fraud_dict
    assert not fraud_dict["labels"].empty


def test_load_and_clean_invalid_name():
    with pytest.raises(ValueError, match="Unknown dataset name"):
        load_and_clean("invalid_dataset_name")
