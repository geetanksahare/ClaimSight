import pandas as pd
import pytest

from claimsight.config import get_settings
from claimsight.inference.flag_anomalies import flag_anomalies, provider_claims


def test_flag_anomalies_output_schema():
    cfg = get_settings()
    sample_path = cfg.paths.demo_dir / "provider_sample.parquet"
    if not sample_path.exists():
        pytest.skip("provider_sample.parquet does not exist yet")

    df = pd.read_parquet(sample_path)
    res = flag_anomalies(df.head(10), top_n=5)

    assert len(res) == 5
    expected_cols = [
        "provider",
        "fraud_prob",
        "anomaly_score",
        "risk_score",
        "flagged",
        "rank",
        "top_reasons",
        "n_claims",
        "reimb_sum",
    ]
    for col in expected_cols:
        assert col in res.columns

    # Ranks should be 1 to 5 in ascending order
    assert list(res["rank"]) == [1, 2, 3, 4, 5]

    # Scores should be sorted descending by risk_score
    risk_scores = res["risk_score"].tolist()
    assert risk_scores == sorted(risk_scores, reverse=True)

    # Values within probability / score bounds [0, 1]
    for _, row in res.iterrows():
        assert 0.0 <= row["fraud_prob"] <= 1.0
        assert 0.0 <= row["anomaly_score"] <= 1.0
        assert 0.0 <= row["risk_score"] <= 1.0
        assert isinstance(row["top_reasons"], list)
        assert len(row["top_reasons"]) > 0


def test_provider_claims():
    claims = pd.DataFrame(
        {
            "ClaimID": ["CLM1", "CLM2", "CLM3"],
            "Provider": ["PRV1", "PRV2", "PRV1"],
            "InscClaimAmtReimbursed": [100.0, 500.0, 2000.0],
        }
    )

    prv1_claims = provider_claims("PRV1", claims)
    assert len(prv1_claims) == 2
    # Check sorted descending by reimbursed amount
    assert prv1_claims.iloc[0]["ClaimID"] == "CLM3"
    assert prv1_claims.iloc[0]["InscClaimAmtReimbursed"] == 2000.0
    assert prv1_claims.iloc[1]["ClaimID"] == "CLM1"
