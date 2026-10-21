from typing import Any

import pandas as pd

from claimsight.config import Settings
from claimsight.logging_config import get_logger

logger = get_logger(__name__)

CHRONIC_CONDITIONS = [
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


def build_member_features(
    data: dict[str, pd.DataFrame],
    cfg: Settings,
) -> pd.DataFrame:
    """Construct longitudinal member feature table from multi-year claims records."""
    bene = data["beneficiary"].copy()
    ip = data.get("inpatient", pd.DataFrame())
    op = data.get("outpatient", pd.DataFrame())
    car_agg = data.get("carrier_agg", pd.DataFrame())

    # 1. Beneficiary features per member and year
    # Age at December 31 of current year in whole years
    bene["year_end"] = pd.to_datetime(bene["year"].astype(str) + "-12-31")
    bene["age_at_year_end"] = (
        ((bene["year_end"] - bene["BENE_BIRTH_DT"]).dt.days / 365.25).fillna(65.0).astype(int)
    )

    present_chronic = [c for c in CHRONIC_CONDITIONS if c in bene.columns]
    bene["chronic_count"] = bene[present_chronic].sum(axis=1)
    bene["hi_months"] = bene["BENE_HI_CVRAGE_TOT_MONS"]
    bene["smi_months"] = bene["BENE_SMI_CVRAGE_TOT_MONS"]
    bene["cost_year"] = bene["MEDREIMB_IP"] + bene["MEDREIMB_OP"] + bene["MEDREIMB_CAR"]

    # 2. Aggregate claims per member and year
    if not ip.empty and "CLM_UTLZTN_DAY_CNT" in ip.columns:
        ip_agg = ip.groupby(["DESYNPUF_ID", "year"], as_index=False).agg(
            ip_claims=("CLM_ID", "count"),
            ip_days=("CLM_UTLZTN_DAY_CNT", "sum"),
            ip_pmt=("CLM_PMT_AMT", "sum"),
        )
    else:
        ip_agg = pd.DataFrame(columns=["DESYNPUF_ID", "year", "ip_claims", "ip_days", "ip_pmt"])

    if not op.empty and "CLM_PMT_AMT" in op.columns:
        op_agg = op.groupby(["DESYNPUF_ID", "year"], as_index=False).agg(
            op_claims=("CLM_ID", "count"),
            op_pmt=("CLM_PMT_AMT", "sum"),
        )
    else:
        op_agg = pd.DataFrame(columns=["DESYNPUF_ID", "year", "op_claims", "op_pmt"])

    # Merge claims onto beneficiary table
    merged = bene.merge(ip_agg, on=["DESYNPUF_ID", "year"], how="left")
    merged = merged.merge(op_agg, on=["DESYNPUF_ID", "year"], how="left")
    if not car_agg.empty:
        merged = merged.merge(car_agg, on=["DESYNPUF_ID", "year"], how="left")
    else:
        merged["carrier_claims"] = 0
        merged["carrier_pmt"] = 0.0

    merged["ip_claims"] = merged["ip_claims"].fillna(0).astype(int)
    merged["ip_days"] = merged["ip_days"].fillna(0).astype(int)
    merged["ip_pmt"] = merged["ip_pmt"].fillna(0.0)
    merged["op_claims"] = merged["op_claims"].fillna(0).astype(int)
    merged["op_pmt"] = merged["op_pmt"].fillna(0.0)
    merged["carrier_claims"] = merged["carrier_claims"].fillna(0).astype(int)
    merged["carrier_pmt"] = merged["carrier_pmt"].fillna(0.0)

    # 3. Longitudinal Pairs: Base Year t in (2008, 2009) predicting Year t+1
    # Create lookup map for year t+1 costs and eligibility
    by_member_year: dict[tuple[str, int], dict[str, Any]] = {}
    for r in merged.to_dict(orient="records"):
        by_member_year[(r["DESYNPUF_ID"], int(r["year"]))] = r

    eligible_rows = []
    filtered_death = 0
    filtered_coverage = 0

    for m_id in merged["DESYNPUF_ID"].unique():
        for base_t in [2008, 2009]:
            target_t = base_t + 1
            if (m_id, base_t) not in by_member_year or (m_id, target_t) not in by_member_year:
                continue

            r_base = by_member_year[(m_id, base_t)]
            r_target = by_member_year[(m_id, target_t)]

            # Eligibility filter 1: alive through end of t+1
            death_dt = r_base["BENE_DEATH_DT"]
            if pd.notna(death_dt):
                target_year_end = pd.Timestamp(f"{target_t}-12-31")
                if death_dt <= target_year_end:
                    filtered_death += 1
                    continue

            # Eligibility filter 2: 12 months continuous HI and SMI in both years
            if not (
                r_base["hi_months"] == 12
                and r_base["smi_months"] == 12
                and r_target["hi_months"] == 12
                and r_target["smi_months"] == 12
            ):
                filtered_coverage += 1
                continue

            prior_tot_claims = r_base["ip_claims"] + r_base["op_claims"] + r_base["carrier_claims"]
            prior_tot_cost = r_base["cost_year"]
            cost_per_clm = prior_tot_cost / max(prior_tot_claims, 1)

            # Prior cost change (cost_t - cost_t-1) when base_t == 2009
            has_hist = 0
            cost_chg = 0.0
            if base_t == 2009 and (m_id, 2008) in by_member_year:
                has_hist = 1
                cost_chg = prior_tot_cost - by_member_year[(m_id, 2008)]["cost_year"]

            row = {
                "member_id": m_id,
                "base_year": base_t,
                "age_at_year_end": r_base["age_at_year_end"],
                "sex": r_base["BENE_SEX_IDENT_CD"],
                "race": str(r_base["BENE_RACE_CD"]),
                "esrd": r_base["BENE_ESRD_IND"],
                **{c: r_base[c] for c in present_chronic},
                "chronic_count": r_base["chronic_count"],
                "prior_cost_ip": r_base["MEDREIMB_IP"],
                "prior_cost_op": r_base["MEDREIMB_OP"],
                "prior_cost_car": r_base["MEDREIMB_CAR"],
                "prior_total_cost": prior_tot_cost,
                "prior_total_claims": prior_tot_claims,
                "cost_per_claim": cost_per_clm,
                "cost_change": cost_chg,
                "has_history": has_hist,
                "ip_days": r_base["ip_days"],
                "ip_claims": r_base["ip_claims"],
                "op_claims": r_base["op_claims"],
                "carrier_claims": r_base["carrier_claims"],
                "next_year_cost": r_target["cost_year"],
            }
            eligible_rows.append(row)

    logger.info(
        "Eligibility filter results - Excluded due to mortality: %d, Excluded due to partial coverage: %d, Final rows: %d",
        filtered_death,
        filtered_coverage,
        len(eligible_rows),
    )

    feature_df = pd.DataFrame(eligible_rows)
    return feature_df


def label_high_cost(
    y_train: pd.Series,
    y_all: pd.Series,
    q: float = 0.90,
) -> tuple[float, pd.Series]:
    """Calculate high-cost cutoff on training target and apply to full series."""
    threshold = float(y_train.quantile(q))
    labels = (y_all >= threshold).astype(int)
    return threshold, labels


def split_member_xy(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate member features and next_year_cost target."""
    if "next_year_cost" not in df.columns:
        raise ValueError("Target 'next_year_cost' not found in DataFrame")
    drop_cols = ["member_id", "base_year", "next_year_cost"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns]).copy()
    y = df["next_year_cost"].copy()
    return X, y
