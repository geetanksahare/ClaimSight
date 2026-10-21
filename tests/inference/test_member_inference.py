import pandas as pd

from claimsight.inference.score_member_risk import _assign_tier, score_member_risk


def test_risk_tier_boundary_assignment():
    cutoffs = [2000.0, 5000.0, 10000.0]

    assert _assign_tier(1500.0, cutoffs) == "Low"
    assert _assign_tier(2000.0, cutoffs) == "Moderate"
    assert _assign_tier(4999.0, cutoffs) == "Moderate"
    assert _assign_tier(5000.0, cutoffs) == "High"
    assert _assign_tier(9999.0, cutoffs) == "High"
    assert _assign_tier(10000.0, cutoffs) == "Very High"
    assert _assign_tier(50000.0, cutoffs) == "Very High"


def test_score_member_risk_dict_output():
    sample_member = {
        "member_id": "MEM_TEST",
        "age": 70,
        "sex": "female",
        "race": "1",
        "esrd": 0,
        "chronic_flags": {
            "SP_DIABETES": 1,
            "SP_CHF": 1,
        },
        "prior_cost_ip": 1500.0,
        "prior_cost_op": 400.0,
        "prior_cost_car": 300.0,
        "ip_claims": 1,
        "op_claims": 2,
        "carrier_claims": 3,
        "ip_days": 3,
        "has_history": 1,
        "cost_change": 100.0,
    }

    res = score_member_risk(sample_member)

    assert "expected_cost" in res
    assert "prob_high_cost" in res
    assert "risk_tier" in res
    assert res["risk_tier"] in ["Low", "Moderate", "High", "Very High"]
    assert 0.0 <= res["prob_high_cost"] <= 1.0
    assert "top_drivers" in res
    assert len(res["top_drivers"]) == 3


def test_score_member_risk_dataframe_output():
    sample_df = pd.DataFrame(
        [
            {
                "member_id": "MEM_01",
                "age": 66,
                "sex": "male",
                "race": "1",
                "esrd": 0,
                "prior_cost_ip": 0.0,
                "prior_cost_op": 100.0,
                "prior_cost_car": 200.0,
                "ip_claims": 0,
                "op_claims": 1,
                "carrier_claims": 2,
                "ip_days": 0,
                "has_history": 0,
                "cost_change": 0.0,
            }
        ]
    )

    out_df = score_member_risk(sample_df)

    assert "expected_cost" in out_df.columns
    assert "prob_high_cost" in out_df.columns
    assert "risk_tier" in out_df.columns
    assert len(out_df) == 1
