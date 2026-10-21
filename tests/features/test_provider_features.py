import pandas as pd
import pytest

from claimsight.config import get_settings
from claimsight.features.provider_features import build_provider_features


@pytest.fixture
def sample_claims_data():
    inpatient = pd.DataFrame(
        {
            "ClaimID": ["CLM_IP_1", "CLM_IP_2"],
            "BeneID": ["BENE_1", "BENE_2"],
            "ClaimStartDt": ["2009-01-01", "2009-02-01"],
            "ClaimEndDt": ["2009-01-05", "2009-02-10"],
            "Provider": ["PRV_001", "PRV_001"],
            "InscClaimAmtReimbursed": [5000.0, 3000.0],
            "DeductibleAmtPaid": [1000.0, 500.0],
            "AdmissionDt": ["2009-01-01", "2009-02-01"],
            "DischargeDt": ["2009-01-05", "2009-02-10"],
            "AttendingPhysician": ["PHY_A", "PHY_A"],
            "OperatingPhysician": ["PHY_B", None],
            "OtherPhysician": [None, None],
            "ClmDiagnosisCode_1": ["4019", "25000"],
            "ClmDiagnosisCode_2": [None, None],
            "ClmProcedureCode_1": ["8154", None],
        }
    )

    outpatient = pd.DataFrame(
        {
            "ClaimID": ["CLM_OP_1"],
            "BeneID": ["BENE_1"],
            "ClaimStartDt": ["2009-01-01"],  # Same start date as CLM_IP_1 for BENE_1
            "ClaimEndDt": ["2009-01-01"],
            "Provider": ["PRV_001"],
            "InscClaimAmtReimbursed": [200.0],
            "DeductibleAmtPaid": [50.0],
            "AdmissionDt": [None],
            "DischargeDt": [None],
            "AttendingPhysician": ["PHY_A"],
            "OperatingPhysician": [None],
            "OtherPhysician": [None],
            "ClmDiagnosisCode_1": ["4019"],
            "ClmDiagnosisCode_2": [None],
            "ClmProcedureCode_1": [None],
        }
    )

    beneficiary = pd.DataFrame(
        {
            "BeneID": ["BENE_1", "BENE_2"],
            "DOB": ["1940-01-01", "1950-05-15"],
            "DOD": [None, None],
            "Gender": [1, 2],
            "Race": [1, 1],
            "RenalDiseaseIndicator": ["0", "1"],
            "State": [1, 1],
            "County": [10, 10],
            "ChronicCond_Alzheimer": [1, 0],
            "ChronicCond_Heartfailure": [0, 1],
            "ChronicCond_KidneyDisease": [0, 1],
            "ChronicCond_Cancer": [0, 0],
            "ChronicCond_ObstrPulmonary": [0, 0],
            "ChronicCond_Depression": [0, 0],
            "ChronicCond_Diabetes": [1, 1],
            "ChronicCond_IschemicHeart": [0, 0],
            "ChronicCond_Osteoporasis": [0, 0],
            "ChronicCond_rheumatoidarthritis": [0, 0],
            "ChronicCond_stroke": [0, 0],
            "IPAnnualReimbursementAmt": [8000.0, 0.0],
            "IPAnnualDeductibleAmt": [1500.0, 0.0],
            "OPAnnualReimbursementAmt": [200.0, 0.0],
            "OPAnnualDeductibleAmt": [50.0, 0.0],
            "NoOfMonths_PartACov": [12, 12],
            "NoOfMonths_PartBCov": [12, 12],
        }
    )

    labels = pd.DataFrame(
        {
            "Provider": ["PRV_001"],
            "PotentialFraud": ["Yes"],
        }
    )

    return {
        "inpatient": inpatient,
        "outpatient": outpatient,
        "beneficiary": beneficiary,
        "labels": labels,
    }


def test_build_provider_features(sample_claims_data):
    cfg = get_settings()
    feat_df = build_provider_features(sample_claims_data, cfg)

    assert len(feat_df) == 1
    row = feat_df.iloc[0]

    assert row["Provider"] == "PRV_001"
    assert row["n_claims"] == 3
    assert row["n_inpatient"] == 2
    assert row["n_outpatient"] == 1
    assert pytest.approx(row["inpatient_share"], 0.01) == 2 / 3
    assert row["n_unique_beneficiaries"] == 2
    assert pytest.approx(row["claims_per_beneficiary"], 0.01) == 1.5

    # Financial sums: 5000 + 3000 + 200 = 8200
    assert pytest.approx(row["reimb_sum"], 0.01) == 8200.0
    assert pytest.approx(row["reimb_mean"], 0.01) == 8200.0 / 3

    # Physician concentration: PHY_A is attending for all 3 claims
    assert row["top_attending_claim_share"] == 1.0
    assert row["attending_missing_share"] == 0.0

    # Same day multiple claims: BENE_1 had 2 claims on 2009-01-01
    assert row["share_claims_same_day_multiple"] > 0

    # Labels joined
    assert row["is_fraud"] == 1
