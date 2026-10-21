from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from claimsight.config import get_settings
from claimsight.data.schemas import (
    cost_clean_schema,
    fraud_labels_schema,
    synpuf_beneficiary_schema,
    synpuf_carrier_agg_schema,
    synpuf_inpatient_schema,
    synpuf_outpatient_schema,
)
from claimsight.logging_config import get_logger

logger = get_logger(__name__)


def _clean_cost(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Cost dataset file not found: {path}")

    df = pd.read_csv(path)
    logger.info("Loaded raw cost data from %s: %d rows, %d columns", path, len(df), len(df.columns))

    cleaned_cols = [col.strip().lower() for col in df.columns]
    expected_cols = ["age", "sex", "bmi", "children", "smoker", "region", "charges"]
    if cleaned_cols != expected_cols:
        raise ValueError(f"Cost column mismatch. Expected {expected_cols}, got {cleaned_cols}")
    df.columns = cleaned_cols

    for col in ["sex", "smoker", "region"]:
        df[col] = df[col].astype(str).str.strip().str.lower()

    df["age"] = df["age"].astype(int)
    df["bmi"] = df["bmi"].astype(float)
    df["children"] = df["children"].astype(int)
    df["charges"] = df["charges"].astype(float)

    if df.isna().any().any():
        raise ValueError("Cost dataset contains unexpected missing values")

    initial_len = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    dropped = initial_len - len(df)
    logger.info("Dropped %d exact duplicate row(s); remaining rows: %d", dropped, len(df))

    df["row_id"] = np.arange(len(df), dtype=np.int64)

    skewness = float(df["charges"].skew())
    logger.info("Charges target skewness: %.4f (retaining high-cost outliers)", skewness)

    validated_df = cost_clean_schema.validate(df)
    return validated_df


def _clean_synpuf(raw_dir: Path) -> dict[str, pd.DataFrame]:
    if not raw_dir.exists():
        raise FileNotFoundError(f"SynPUF raw directory not found: {raw_dir}")

    # Beneficiary summary files across years 2008, 2009, 2010
    bene_files = sorted(raw_dir.glob("*Beneficiary_Summary_File_Sample_1*.csv"))
    if not bene_files:
        bene_files = sorted(raw_dir.glob("*beneficiary*.csv"))

    if not bene_files:
        raise FileNotFoundError(f"No SynPUF beneficiary files found in {raw_dir}")

    bene_dfs = []
    for bf in bene_files:
        year_str = None
        for y in ["2008", "2009", "2010"]:
            if y in bf.name:
                year_str = int(y)
                break
        if year_str is None:
            continue
        sub_df = pd.read_csv(bf, low_memory=False)
        sub_df["year"] = year_str
        bene_dfs.append(sub_df)

    if not bene_dfs:
        raise ValueError("Could not parse beneficiary years from file names")

    bene = pd.concat(bene_dfs, ignore_index=True)
    logger.info("Loaded concatenated beneficiary records: %d rows", len(bene))

    expected_bene = [
        "DESYNPUF_ID",
        "BENE_BIRTH_DT",
        "BENE_DEATH_DT",
        "BENE_SEX_IDENT_CD",
        "BENE_RACE_CD",
        "BENE_ESRD_IND",
        "SP_STATE_CODE",
        "BENE_HI_CVRAGE_TOT_MONS",
        "BENE_SMI_CVRAGE_TOT_MONS",
        "MEDREIMB_IP",
        "MEDREIMB_OP",
        "MEDREIMB_CAR",
    ]
    for col in expected_bene:
        if col not in bene.columns:
            raise ValueError(f"Missing expected beneficiary column: {col}")

    chronic_cols = [
        c
        for c in [
            "SP_ALZHDMTA",
            "SP_CHF",
            "SP_CHRNKIDN",
            "SP_CNCR",
            "SP_COPD",
            "SP_DEPRESSN",
            "SP_DIABETES",
            "SP_ISCHMCHT",
            "SP_OSTEOPRS",
            "SP_RA_OA",
            "SP_STRKETIA",
        ]
        if c in bene.columns
    ]

    keep_bene_cols = expected_bene + chronic_cols + ["year"]
    bene = bene[keep_bene_cols].copy()

    bene["BENE_BIRTH_DT"] = pd.to_datetime(
        bene["BENE_BIRTH_DT"].astype(str), format="%Y%m%d", errors="coerce"
    )
    bene["BENE_DEATH_DT"] = pd.to_datetime(
        bene["BENE_DEATH_DT"].astype(str), format="%Y%m%d", errors="coerce"
    )

    for c in chronic_cols:
        bene[c] = bene[c].map({1: 1, 2: 0}).fillna(0).astype(int)

    bene["BENE_ESRD_IND"] = (bene["BENE_ESRD_IND"].astype(str).str.upper() == "Y").astype(int)
    bene["BENE_SEX_IDENT_CD"] = bene["BENE_SEX_IDENT_CD"].map(
        {1: "male", 2: "female", "1": "male", "2": "female"}
    )
    bene["BENE_RACE_CD"] = bene["BENE_RACE_CD"].astype(str)
    bene["SP_STATE_CODE"] = bene["SP_STATE_CODE"].astype(str)

    for num_col in [
        "BENE_HI_CVRAGE_TOT_MONS",
        "BENE_SMI_CVRAGE_TOT_MONS",
        "MEDREIMB_IP",
        "MEDREIMB_OP",
        "MEDREIMB_CAR",
    ]:
        bene[num_col] = pd.to_numeric(bene[num_col], errors="coerce").fillna(0)

    bene = synpuf_beneficiary_schema.validate(bene)

    # Inpatient
    ip_files = sorted(raw_dir.glob("*Inpatient_Claims_Sample_1*.csv")) or sorted(
        raw_dir.glob("*inpatient*.csv")
    )
    if not ip_files:
        raise FileNotFoundError(f"Missing SynPUF inpatient files in {raw_dir}")
    ip = pd.concat([pd.read_csv(f, low_memory=False) for f in ip_files], ignore_index=True)
    expected_ip = [
        "DESYNPUF_ID",
        "CLM_ID",
        "CLM_FROM_DT",
        "CLM_THRU_DT",
        "CLM_PMT_AMT",
        "CLM_UTLZTN_DAY_CNT",
    ]
    for col in expected_ip:
        if col not in ip.columns:
            raise ValueError(f"Missing expected inpatient column: {col}")
    ip = ip[expected_ip].copy()
    ip["CLM_FROM_DT"] = pd.to_datetime(
        ip["CLM_FROM_DT"].astype(str), format="%Y%m%d", errors="coerce"
    )
    ip["CLM_THRU_DT"] = pd.to_datetime(
        ip["CLM_THRU_DT"].astype(str), format="%Y%m%d", errors="coerce"
    )
    ip["year"] = ip["CLM_FROM_DT"].dt.year.fillna(2008).astype(int)
    ip["CLM_PMT_AMT"] = pd.to_numeric(ip["CLM_PMT_AMT"], errors="coerce").fillna(0.0)
    ip["CLM_UTLZTN_DAY_CNT"] = pd.to_numeric(ip["CLM_UTLZTN_DAY_CNT"], errors="coerce").fillna(0.0)
    ip = synpuf_inpatient_schema.validate(ip)

    # Outpatient
    op_files = sorted(raw_dir.glob("*Outpatient_Claims_Sample_1*.csv")) or sorted(
        raw_dir.glob("*outpatient*.csv")
    )
    if not op_files:
        raise FileNotFoundError(f"Missing SynPUF outpatient files in {raw_dir}")
    op = pd.concat([pd.read_csv(f, low_memory=False) for f in op_files], ignore_index=True)
    expected_op = ["DESYNPUF_ID", "CLM_ID", "CLM_FROM_DT", "CLM_PMT_AMT"]
    for col in expected_op:
        if col not in op.columns:
            raise ValueError(f"Missing expected outpatient column: {col}")
    op = op[expected_op].copy()
    op["CLM_FROM_DT"] = pd.to_datetime(
        op["CLM_FROM_DT"].astype(str), format="%Y%m%d", errors="coerce"
    )
    op["year"] = op["CLM_FROM_DT"].dt.year.fillna(2008).astype(int)
    op["CLM_PMT_AMT"] = pd.to_numeric(op["CLM_PMT_AMT"], errors="coerce").fillna(0.0)
    op = synpuf_outpatient_schema.validate(op)

    # Carrier files in chunks
    carrier_files = sorted(raw_dir.glob("*Carrier_Claims_Sample_1*.csv")) or sorted(
        raw_dir.glob("*carrier*.csv")
    )
    if not carrier_files:
        raise FileNotFoundError(f"Missing SynPUF carrier files in {raw_dir}")

    chunk_aggs = []
    for cf in carrier_files:
        sample_head = pd.read_csv(cf, nrows=2)
        base_cols = ["DESYNPUF_ID", "CLM_ID", "CLM_FROM_DT"]
        line_cols = [c for c in sample_head.columns if c.startswith("LINE_NCH_PMT_AMT_")]
        use_cols = base_cols + line_cols
        for chunk in pd.read_csv(cf, chunksize=500_000, usecols=use_cols, low_memory=False):
            dates = pd.to_datetime(
                chunk["CLM_FROM_DT"].astype(str), format="%Y%m%d", errors="coerce"
            )
            chunk["year"] = dates.dt.year.fillna(2008).astype(int)
            if line_cols:
                chunk["claim_pmt"] = (
                    chunk[line_cols].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1)
                )
            else:
                chunk["claim_pmt"] = 0.0

            agg = (
                chunk.groupby(["DESYNPUF_ID", "year"])
                .agg(
                    carrier_claims=("CLM_ID", "nunique"),
                    carrier_pmt=("claim_pmt", "sum"),
                )
                .reset_index()
            )
            chunk_aggs.append(agg)

    if chunk_aggs:
        all_carrier = pd.concat(chunk_aggs, ignore_index=True)
        carrier_agg = all_carrier.groupby(["DESYNPUF_ID", "year"], as_index=False).agg(
            {"carrier_claims": "sum", "carrier_pmt": "sum"}
        )
    else:
        carrier_agg = pd.DataFrame(columns=["DESYNPUF_ID", "year", "carrier_claims", "carrier_pmt"])

    carrier_agg = synpuf_carrier_agg_schema.validate(carrier_agg)

    return {
        "beneficiary": bene,
        "inpatient": ip,
        "outpatient": op,
        "carrier_agg": carrier_agg,
    }


def _clean_fraud(raw_dir: Path) -> dict[str, pd.DataFrame]:
    if not raw_dir.exists():
        raise FileNotFoundError(f"Fraud raw directory not found: {raw_dir}")

    bene_files = list(raw_dir.glob("Train_Beneficiarydata*.csv")) or list(
        raw_dir.glob("*beneficiary*.csv")
    )
    inpatient_files = list(raw_dir.glob("Train_Inpatientdata*.csv")) or list(
        raw_dir.glob("*inpatient*.csv")
    )
    outpatient_files = list(raw_dir.glob("Train_Outpatientdata*.csv")) or list(
        raw_dir.glob("*outpatient*.csv")
    )
    labels_files = list(raw_dir.glob("Train-*.csv")) or list(raw_dir.glob("*fraud*.csv"))

    if not (bene_files and inpatient_files and outpatient_files and labels_files):
        raise FileNotFoundError(f"One or more required fraud files missing in {raw_dir}")

    bene = pd.read_csv(bene_files[0])
    ip = pd.read_csv(inpatient_files[0])
    op = pd.read_csv(outpatient_files[0])
    labels = pd.read_csv(labels_files[0])

    expected_labels = ["Provider", "PotentialFraud"]
    for c in expected_labels:
        if c not in labels.columns:
            raise ValueError(f"Missing column in fraud labels: {c}")

    labels = labels.rename(columns={"PotentialFraud": "is_fraud"}).copy()
    labels["is_fraud"] = (labels["is_fraud"].astype(str).str.strip().str.lower() == "yes").astype(
        int
    )
    labels = fraud_labels_schema.validate(labels)

    labeled_providers = set(labels["Provider"].unique())
    ip_initial, op_initial = len(ip), len(op)

    ip = ip[ip["Provider"].isin(labeled_providers)].copy()
    op = op[op["Provider"].isin(labeled_providers)].copy()

    logger.info(
        "Dropped %d inpatient and %d outpatient claims belonging to unlabeled providers",
        ip_initial - len(ip),
        op_initial - len(op),
    )

    ip["claim_type"] = "inpatient"
    op["claim_type"] = "outpatient"

    # Date and chronic code parsing
    for cond_col in [c for c in bene.columns if c.startswith("ChronicCond_")]:
        bene[cond_col] = bene[cond_col].map({1: 1, 2: 0}).fillna(0).astype(int)

    if "RenalDiseaseIndicator" in bene.columns:
        bene["RenalDiseaseIndicator"] = (
            bene["RenalDiseaseIndicator"].astype(str).str.strip().str.upper() == "Y"
        ).astype(int)

    for d_col in ["DOB", "DOD"]:
        if d_col in bene.columns:
            bene[d_col] = pd.to_datetime(bene[d_col], errors="coerce")

    for dt_col in ["ClaimStartDt", "ClaimEndDt", "AdmissionDt", "DischargeDt"]:
        if dt_col in ip.columns:
            ip[dt_col] = pd.to_datetime(ip[dt_col], errors="coerce")
        if dt_col in op.columns:
            op[dt_col] = pd.to_datetime(op[dt_col], errors="coerce")

    return {
        "beneficiary": bene,
        "inpatient": ip,
        "outpatient": op,
        "labels": labels,
    }


def load_and_clean(
    dataset: Literal["cost", "synpuf", "fraud"],
    raw_dir: Path | None = None,
    save: bool = True,
) -> pd.DataFrame | dict[str, pd.DataFrame]:
    cfg = get_settings()
    base_raw = raw_dir if raw_dir is not None else cfg.paths.raw_dir

    if dataset == "cost":
        cost_path = (
            base_raw / "cost" / "insurance.csv"
            if (base_raw / "cost").exists()
            else base_raw / "insurance.csv"
        )
        cleaned = _clean_cost(cost_path)
        if save:
            cfg.paths.processed_dir.mkdir(parents=True, exist_ok=True)
            out_file = cfg.paths.processed_dir / "cost.parquet"
            cleaned.to_parquet(out_file, index=False)
            logger.info("Saved cleaned cost table to %s", out_file)
        return cleaned

    if dataset == "synpuf":
        synpuf_dir = base_raw / "synpuf" if (base_raw / "synpuf").exists() else base_raw
        cleaned = _clean_synpuf(synpuf_dir)
        if save:
            cfg.paths.processed_dir.mkdir(parents=True, exist_ok=True)
            for k, df in cleaned.items():
                out_file = cfg.paths.processed_dir / f"synpuf_{k}.parquet"
                df.to_parquet(out_file, index=False)
                logger.info("Saved %s to %s", k, out_file)
        return cleaned

    if dataset == "fraud":
        fraud_dir = base_raw / "fraud" if (base_raw / "fraud").exists() else base_raw
        cleaned = _clean_fraud(fraud_dir)
        if save:
            cfg.paths.processed_dir.mkdir(parents=True, exist_ok=True)
            for k, df in cleaned.items():
                out_file = cfg.paths.processed_dir / f"fraud_{k}.parquet"
                df.to_parquet(out_file, index=False)
                logger.info("Saved %s to %s", k, out_file)
        return cleaned

    raise ValueError(f"Unknown dataset name: {dataset}")
