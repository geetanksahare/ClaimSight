from pathlib import Path

import numpy as np
import pandas as pd

from claimsight.config import get_settings


def generate_synthetic_fraud(
    n_providers: int = 1200, n_beneficiaries: int = 4000, seed: int = 42
) -> Path:
    """Generate realistic synthetic Healthcare Provider Fraud dataset in data/raw/fraud/."""
    rng = np.random.RandomState(seed)
    cfg = get_settings()
    raw_dir = cfg.paths.raw_dir / "fraud"
    raw_dir.mkdir(parents=True, exist_ok=True)

    providers = [f"PRV{i:05d}" for i in range(1, n_providers + 1)]
    beneficiaries = [f"BENE_{i:06d}" for i in range(1, n_beneficiaries + 1)]

    # 1. Labels (~9% fraudulent providers as per spec)
    is_fraud_mask = rng.uniform(size=n_providers) < 0.091
    labels_df = pd.DataFrame(
        {
            "Provider": providers,
            "PotentialFraud": ["Yes" if f else "No" for f in is_fraud_mask],
        }
    )
    labels_df.to_csv(raw_dir / "Train-1542865627584.csv", index=False)

    fraud_provider_set = set(labels_df[labels_df["PotentialFraud"] == "Yes"]["Provider"])

    # 2. Beneficiaries
    chronic_names = [
        "ChronicCond_Alzheimer",
        "ChronicCond_Heartfailure",
        "ChronicCond_KidneyDisease",
        "ChronicCond_Cancer",
        "ChronicCond_ObstrPulmonary",
        "ChronicCond_Depression",
        "ChronicCond_Diabetes",
        "ChronicCond_IschemicHeart",
        "ChronicCond_Osteoporasis",
        "ChronicCond_rheumatoidarthritis",
        "ChronicCond_stroke",
    ]
    bene_dob = [
        f"{rng.randint(1925, 1950)}-{rng.randint(1, 13):02d}-{rng.randint(1, 28):02d}"
        for _ in range(n_beneficiaries)
    ]
    bene_df = pd.DataFrame(
        {
            "BeneID": beneficiaries,
            "DOB": bene_dob,
            "DOD": ["" if rng.uniform() > 0.02 else "2009-08-15" for _ in range(n_beneficiaries)],
            "Gender": rng.choice([1, 2], size=n_beneficiaries),
            "Race": rng.choice([1, 2, 3, 5], size=n_beneficiaries),
            "RenalDiseaseIndicator": rng.choice(["0", "Y"], size=n_beneficiaries, p=[0.95, 0.05]),
            "State": rng.randint(1, 55, size=n_beneficiaries),
            "County": rng.randint(1, 100, size=n_beneficiaries),
            "NoOfMonths_PartACov": [12] * n_beneficiaries,
            "NoOfMonths_PartBCov": [12] * n_beneficiaries,
            **{c: rng.choice([1, 2], size=n_beneficiaries, p=[0.25, 0.75]) for c in chronic_names},
            "IPAnnualReimbursementAmt": np.round(rng.exponential(2500, size=n_beneficiaries), 2),
            "IPAnnualDeductibleAmt": np.round(rng.exponential(400, size=n_beneficiaries), 2),
            "OPAnnualReimbursementAmt": np.round(rng.exponential(1200, size=n_beneficiaries), 2),
            "OPAnnualDeductibleAmt": np.round(rng.exponential(150, size=n_beneficiaries), 2),
        }
    )
    bene_df.to_csv(raw_dir / "Train_Beneficiarydata-1542865627584.csv", index=False)

    # 3. Claims (Inpatient & Outpatient)
    physicians = [f"PHY{i:05d}" for i in range(1, 500)]
    diag_codes = [f"D{i:04d}" for i in range(1, 200)]
    proc_codes = [f"P{i:04d}" for i in range(1, 100)]

    ip_claims = []
    op_claims = []
    claim_id_counter = 1

    for prv in providers:
        is_f = prv in fraud_provider_set
        # Fraudulent providers bill substantially higher volumes and higher repeat billing
        n_claims = rng.randint(30, 180) if is_f else rng.randint(5, 50)
        p_inpatient = rng.uniform(0.1, 0.4)

        top_doc = rng.choice(physicians)

        for _ in range(n_claims):
            bene_id = rng.choice(beneficiaries)
            is_ip = rng.uniform() < p_inpatient

            # Fraud patterns: repeat claims on same date, high reimb, high physician concentration
            m = rng.randint(1, 13)
            d = rng.randint(1, 25)
            start_dt = f"2009-{m:02d}-{d:02d}"
            dur = rng.randint(1, 10) if is_ip else rng.randint(0, 3)
            end_dt = f"2009-{m:02d}-{min(d + dur, 28):02d}"

            # Physician assignment
            if is_f and rng.uniform() < 0.75:
                attending = top_doc
            else:
                attending = rng.choice(physicians) if rng.uniform() > 0.05 else ""

            operating = rng.choice(physicians) if rng.uniform() > 0.4 else ""
            other_doc = rng.choice(physicians) if rng.uniform() > 0.6 else ""

            # Amounts
            if is_ip:
                reimb = float(np.exp(rng.normal(9.2 if is_f else 8.5, 0.6)))
                deductible = float(rng.choice([1068, 0, 500]))
            else:
                reimb = float(np.exp(rng.normal(6.5 if is_f else 5.8, 0.7)))
                deductible = float(rng.choice([0, 50, 100]))

            # Diagnosis and procedure codes
            n_diag = rng.randint(1, 8)
            n_proc = rng.randint(0, 4) if is_ip else (1 if rng.uniform() < 0.2 else 0)

            chosen_diags = list(rng.choice(diag_codes, size=n_diag, replace=False))
            chosen_procs = (
                list(rng.choice(proc_codes, size=n_proc, replace=False)) if n_proc > 0 else []
            )

            claim_record = {
                "BeneID": bene_id,
                "ClaimID": f"CLM{claim_id_counter:08d}",
                "ClaimStartDt": start_dt,
                "ClaimEndDt": end_dt,
                "Provider": prv,
                "InscClaimAmtReimbursed": round(reimb, 2),
                "AttendingPhysician": attending,
                "OperatingPhysician": operating,
                "OtherPhysician": other_doc,
                "DeductibleAmtPaid": round(deductible, 2),
                **{
                    f"ClmDiagnosisCode_{i}": (chosen_diags[i - 1] if i <= len(chosen_diags) else "")
                    for i in range(1, 11)
                },
                **{
                    f"ClmProcedureCode_{i}": (chosen_procs[i - 1] if i <= len(chosen_procs) else "")
                    for i in range(1, 7)
                },
            }

            if is_ip:
                claim_record["AdmissionDt"] = start_dt
                claim_record["DischargeDt"] = end_dt
                ip_claims.append(claim_record)
            else:
                op_claims.append(claim_record)

            claim_id_counter += 1

    ip_df = pd.DataFrame(ip_claims)
    op_df = pd.DataFrame(op_claims)

    ip_df.to_csv(raw_dir / "Train_Inpatientdata-1542865627584.csv", index=False)
    op_df.to_csv(raw_dir / "Train_Outpatientdata-1542865627584.csv", index=False)

    return raw_dir


if __name__ == "__main__":
    generate_synthetic_fraud()
    print("Generated synthetic Fraud dataset in data/raw/fraud/")
