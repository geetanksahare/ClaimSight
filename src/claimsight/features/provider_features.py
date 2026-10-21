import numpy as np
import pandas as pd

from claimsight.config import Settings
from claimsight.logging_config import get_logger

logger = get_logger(__name__)


def build_provider_features(
    data: dict[str, pd.DataFrame],
    cfg: Settings,
) -> pd.DataFrame:
    """Aggregate claim and beneficiary records into healthcare provider fraud risk profiles."""
    ip = data["inpatient"].copy()
    op = data["outpatient"].copy()
    bene = data["beneficiary"].copy()
    labels = data.get("labels", pd.DataFrame())

    ip["claim_type"] = "inpatient"
    op["claim_type"] = "outpatient"

    # Harmonize and concatenate claims
    common_cols = [
        "BeneID",
        "ClaimID",
        "ClaimStartDt",
        "ClaimEndDt",
        "Provider",
        "InscClaimAmtReimbursed",
        "AttendingPhysician",
        "OperatingPhysician",
        "OtherPhysician",
        "DeductibleAmtPaid",
        "claim_type",
    ]
    diag_cols = [c for c in ip.columns if c.startswith("ClmDiagnosisCode_")]
    proc_cols = [c for c in ip.columns if c.startswith("ClmProcedureCode_")]

    all_claim_cols = common_cols + diag_cols + proc_cols
    ip_sub = ip[[c for c in all_claim_cols if c in ip.columns]].copy()
    op_sub = op[[c for c in all_claim_cols if c in op.columns]].copy()

    # Inpatient admission/discharge dates for LOS
    if "AdmissionDt" in ip.columns and "DischargeDt" in ip.columns:
        ip_sub["los"] = (
            pd.to_datetime(ip["DischargeDt"], errors="coerce")
            - pd.to_datetime(ip["AdmissionDt"], errors="coerce")
        ).dt.days.fillna(0.0)
    else:
        ip_sub["los"] = 0.0
    op_sub["los"] = 0.0

    claims = pd.concat([ip_sub, op_sub], ignore_index=True)

    # Claim duration
    start_dt = pd.to_datetime(claims["ClaimStartDt"], errors="coerce")
    end_dt = pd.to_datetime(claims["ClaimEndDt"], errors="coerce")
    claims["claim_duration"] = (end_dt - start_dt).dt.days.fillna(0.0).clip(lower=0.0)

    # Coding count per claim
    claims["n_diag_codes"] = claims[diag_cols].notna().sum(axis=1) if diag_cols else 0
    claims["n_proc_codes"] = claims[proc_cols].notna().sum(axis=1) if proc_cols else 0

    # Beneficiary demographics merge
    # Compute age as of 2009-12-31
    if "DOB" in bene.columns:
        bene_dob = pd.to_datetime(bene["DOB"], errors="coerce")
        bene["bene_age"] = ((pd.Timestamp("2009-12-31") - bene_dob).dt.days / 365.25).fillna(65.0)
    else:
        bene["bene_age"] = 65.0

    chronic_cols = [c for c in bene.columns if c.startswith("ChronicCond_")]
    bene["chronic_count"] = bene[chronic_cols].sum(axis=1) if chronic_cols else 0

    # Left-merge beneficiary features onto claims
    bene_merge_cols = ["BeneID", "bene_age", "chronic_count"]
    if "RenalDiseaseIndicator" in bene.columns:
        bene_merge_cols.append("RenalDiseaseIndicator")
    if "NoOfMonths_PartACov" in bene.columns:
        bene_merge_cols.append("NoOfMonths_PartACov")

    claims = claims.merge(bene[bene_merge_cols], on="BeneID", how="left")

    # Repeat billing: claims by same provider & beneficiary on same start date beyond the first
    repeat_counts = claims.groupby(["Provider", "BeneID", "ClaimStartDt"])["ClaimID"].transform(
        "count"
    )
    claims["is_repeat_claim"] = (repeat_counts > 1).astype(int)

    # Provider Aggregations
    records = []
    for prv, g in claims.groupby("Provider"):
        n_claims = len(g)
        n_inpatient = int((g["claim_type"] == "inpatient").sum())
        n_outpatient = n_claims - n_inpatient
        inpatient_share = n_inpatient / n_claims

        n_unique_bene = int(g["BeneID"].nunique())
        claims_per_bene = n_claims / max(n_unique_bene, 1)

        # Money metrics
        reimbs = g["InscClaimAmtReimbursed"].fillna(0.0).to_numpy()
        reimb_sum = float(np.sum(reimbs))
        reimb_mean = float(np.mean(reimbs))
        reimb_median = float(np.median(reimbs))
        reimb_max = float(np.max(reimbs))
        reimb_std = float(np.std(reimbs)) if len(reimbs) > 1 else 0.0
        deductible_mean = float(g["DeductibleAmtPaid"].fillna(0.0).mean())

        # Physician metrics
        attending_series = g["AttendingPhysician"].fillna("").astype(str).str.strip()
        attending_missing = int((attending_series == "").sum())
        attending_valid = attending_series[attending_series != ""]
        n_unique_att = int(attending_valid.nunique())
        top_att_share = (
            float(attending_valid.value_counts().iloc[0] / n_claims)
            if len(attending_valid) > 0
            else 0.0
        )

        operating_series = g["OperatingPhysician"].fillna("").astype(str).str.strip()
        operating_missing = int((operating_series == "").sum())
        n_unique_op = int(operating_series[operating_series != ""].nunique())

        # Durations & coding
        mean_los = (
            float(g[g["claim_type"] == "inpatient"]["los"].mean()) if n_inpatient > 0 else 0.0
        )
        mean_claim_dur = float(g["claim_duration"].mean())

        mean_n_diag = float(g["n_diag_codes"].mean())
        mean_n_proc = float(g["n_proc_codes"].mean())

        # Unique diag codes across all claims of provider
        unique_diags = set()
        for d_col in diag_cols:
            unique_diags.update(g[d_col].dropna().astype(str).unique())
        unique_diags.discard("")
        n_unique_diags = len(unique_diags)

        # Beneficiary mix
        bene_mean_age = float(g["bene_age"].mean())
        bene_chronic_mean = float(g["chronic_count"].mean())
        bene_renal_share = (
            float((g["RenalDiseaseIndicator"] == 1).mean())
            if "RenalDiseaseIndicator" in g.columns
            else 0.0
        )
        bene_mean_partA = (
            float(g["NoOfMonths_PartACov"].mean()) if "NoOfMonths_PartACov" in g.columns else 12.0
        )

        # Repeat billing share
        repeat_share = float(g["is_repeat_claim"].sum() / n_claims)

        records.append(
            {
                "Provider": prv,
                "n_claims": n_claims,
                "n_inpatient": n_inpatient,
                "n_outpatient": n_outpatient,
                "inpatient_share": round(inpatient_share, 4),
                "n_unique_beneficiaries": n_unique_bene,
                "claims_per_beneficiary": round(claims_per_bene, 4),
                "reimb_sum": round(reimb_sum, 2),
                "reimb_mean": round(reimb_mean, 2),
                "reimb_median": round(reimb_median, 2),
                "reimb_max": round(reimb_max, 2),
                "reimb_std": round(reimb_std, 2),
                "deductible_mean": round(deductible_mean, 2),
                "n_unique_attending": n_unique_att,
                "n_unique_operating": n_unique_op,
                "attending_missing_share": round(attending_missing / n_claims, 4),
                "operating_missing_share": round(operating_missing / n_claims, 4),
                "top_attending_claim_share": round(top_att_share, 4),
                "mean_los": round(mean_los, 2),
                "mean_claim_duration": round(mean_claim_dur, 2),
                "mean_n_diagnosis_codes": round(mean_n_diag, 2),
                "mean_n_procedure_codes": round(mean_n_proc, 2),
                "n_unique_diag_codes": n_unique_diags,
                "bene_mean_age": round(bene_mean_age, 2),
                "bene_chronic_mean": round(bene_chronic_mean, 2),
                "bene_renal_share": round(bene_renal_share, 4),
                "bene_mean_partA_months": round(bene_mean_partA, 2),
                "share_claims_same_day_multiple": round(repeat_share, 4),
            }
        )

    provider_df = pd.DataFrame(records)

    # Attach is_fraud if labels present
    if not labels.empty and "Provider" in labels.columns:
        lbl = labels.copy()
        if "is_fraud" not in lbl.columns and "PotentialFraud" in lbl.columns:
            lbl["is_fraud"] = (
                lbl["PotentialFraud"].astype(str).str.lower().isin(["yes", "1"])
            ).astype(int)
        if "is_fraud" in lbl.columns:
            provider_df = provider_df.merge(
                lbl[["Provider", "is_fraud"]], on="Provider", how="inner"
            )
            logger.info(
                "Joined fraud labels: %d providers labeled (fraud rate: %.2f%%)",
                len(provider_df),
                provider_df["is_fraud"].mean() * 100,
            )

    return provider_df


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate provider input features and is_fraud binary label."""
    if "is_fraud" not in df.columns:
        raise ValueError("Label column 'is_fraud' missing from DataFrame")
    X = df.drop(columns=["is_fraud"]).copy()
    y = df["is_fraud"].copy()
    return X, y
