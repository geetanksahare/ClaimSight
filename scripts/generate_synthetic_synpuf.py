from pathlib import Path

import numpy as np
import pandas as pd

from claimsight.config import get_settings


def generate_synthetic_synpuf(n_members: int = 2500, seed: int = 42) -> Path:
    """Generate realistic synthetic CMS SynPUF Sample 1 tables in data/raw/synpuf/."""
    rng = np.random.RandomState(seed)
    cfg = get_settings()
    raw_dir = cfg.paths.raw_dir / "synpuf"
    raw_dir.mkdir(parents=True, exist_ok=True)

    member_ids = [f"BENE_{i:06d}" for i in range(1, n_members + 1)]

    # 1. Beneficiary Demographic Baseline
    birth_years = rng.randint(1925, 1948, size=n_members)
    birth_months = rng.randint(1, 13, size=n_members)
    birth_days = rng.randint(1, 28, size=n_members)
    birth_dates = [f"{y}{m:02d}{d:02d}" for y, m, d in zip(birth_years, birth_months, birth_days)]

    # Mortality: ~3% die in 2010
    death_dates = []
    for _ in range(n_members):
        if rng.uniform() < 0.03:
            d_month = rng.randint(1, 13)
            d_day = rng.randint(1, 28)
            death_dates.append(f"2010{d_month:02d}{d_day:02d}")
        else:
            death_dates.append("")

    sex_codes = rng.choice([1, 2], size=n_members, p=[0.45, 0.55])
    race_codes = rng.choice([1, 2, 3, 5], size=n_members, p=[0.82, 0.11, 0.04, 0.03])
    state_codes = rng.choice([f"{i:02d}" for i in range(1, 55)], size=n_members)
    esrd_ind = rng.choice(["0", "Y"], size=n_members, p=[0.96, 0.04])

    chronic_names = [
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

    # Baseline chronic conditions (1 = Yes, 2 = No in SynPUF)
    baseline_chronic = {
        name: rng.choice([1, 2], size=n_members, p=[0.25, 0.75]) for name in chronic_names
    }

    # Generate 3 yearly beneficiary summary files: 2008, 2009, 2010
    for year in [2008, 2009, 2010]:
        # Coverage months: 94% have continuous full-year coverage (12 months)
        hi_months = np.where(
            rng.uniform(size=n_members) < 0.94, 12, rng.randint(1, 12, size=n_members)
        )
        smi_months = np.where(
            rng.uniform(size=n_members) < 0.94, 12, rng.randint(1, 12, size=n_members)
        )

        # Chronic progression: some members develop conditions in later years
        if year > 2008:
            for name in chronic_names:
                new_cases = (baseline_chronic[name] == 2) & (rng.uniform(size=n_members) < 0.04)
                baseline_chronic[name][new_cases] = 1

        # Costs: correlated with chronic conditions and age
        n_chronic = sum((baseline_chronic[n] == 1).astype(int) for n in chronic_names)
        base_cost = np.exp(rng.normal(6.5 + 0.25 * n_chronic, 0.9, size=n_members))

        # Reimb amounts (SynPUF columns)
        ip_cost = np.where(
            rng.uniform(size=n_members) < 0.18 + 0.05 * n_chronic,
            base_cost * rng.uniform(0.5, 0.8),
            0.0,
        )
        op_cost = np.where(
            rng.uniform(size=n_members) < 0.65, base_cost * rng.uniform(0.1, 0.3), 0.0
        )
        car_cost = np.where(
            rng.uniform(size=n_members) < 0.85, base_cost * rng.uniform(0.1, 0.25), 0.0
        )

        # In 2010, apply death dates for deceased members
        cur_deaths = death_dates if year == 2010 else [""] * n_members

        bene_df = pd.DataFrame(
            {
                "DESYNPUF_ID": member_ids,
                "BENE_BIRTH_DT": birth_dates,
                "BENE_DEATH_DT": cur_deaths,
                "BENE_SEX_IDENT_CD": sex_codes,
                "BENE_RACE_CD": race_codes,
                "BENE_ESRD_IND": esrd_ind,
                "SP_STATE_CODE": state_codes,
                "BENE_HI_CVRAGE_TOT_MONS": hi_months,
                "BENE_SMI_CVRAGE_TOT_MONS": smi_months,
                **{name: baseline_chronic[name] for name in chronic_names},
                "MEDREIMB_IP": np.round(ip_cost, 2),
                "MEDREIMB_OP": np.round(op_cost, 2),
                "MEDREIMB_CAR": np.round(car_cost, 2),
            }
        )

        out_bene = raw_dir / f"DE1_0_{year}_Beneficiary_Summary_File_Sample_1.csv"
        bene_df.to_csv(out_bene, index=False)

    # 2. Inpatient Claims
    ip_records = []
    clm_counter = 1
    for m_id, n_c in zip(member_ids, n_chronic):
        # Members with hospital admissions
        n_admissions = rng.poisson(0.15 + 0.05 * n_c)
        for _ in range(n_admissions):
            y = rng.choice([2008, 2009, 2010])
            m = rng.randint(1, 13)
            d = rng.randint(1, 20)
            los = rng.randint(1, 14)
            from_dt = f"{y}{m:02d}{d:02d}"
            thru_dt = f"{y}{m:02d}{d + los:02d}"
            pmt = round(float(np.exp(rng.normal(8.5, 0.8))), 2)

            ip_records.append(
                {
                    "DESYNPUF_ID": m_id,
                    "CLM_ID": f"IP_CLM_{clm_counter:07d}",
                    "CLM_FROM_DT": from_dt,
                    "CLM_THRU_DT": thru_dt,
                    "CLM_PMT_AMT": pmt,
                    "CLM_UTLZTN_DAY_CNT": los,
                }
            )
            clm_counter += 1

    ip_df = pd.DataFrame(ip_records)
    ip_df.to_csv(raw_dir / "DE1_0_2008_to_2010_Inpatient_Claims_Sample_1.csv", index=False)

    # 3. Outpatient Claims
    op_records = []
    op_counter = 1
    for m_id, n_c in zip(member_ids, n_chronic):
        n_visits = rng.poisson(1.2 + 0.3 * n_c)
        for _ in range(n_visits):
            y = rng.choice([2008, 2009, 2010])
            m = rng.randint(1, 13)
            d = rng.randint(1, 28)
            from_dt = f"{y}{m:02d}{d:02d}"
            pmt = round(float(np.exp(rng.normal(5.8, 0.9))), 2)

            op_records.append(
                {
                    "DESYNPUF_ID": m_id,
                    "CLM_ID": f"OP_CLM_{op_counter:07d}",
                    "CLM_FROM_DT": from_dt,
                    "CLM_PMT_AMT": pmt,
                }
            )
            op_counter += 1

    op_df = pd.DataFrame(op_records)
    op_df.to_csv(raw_dir / "DE1_0_2008_to_2010_Outpatient_Claims_Sample_1.csv", index=False)

    # 4. Carrier Claims (Carrier 1 & Carrier 2)
    car_records = []
    car_counter = 1
    for m_id in member_ids:
        n_car = rng.poisson(2.5)
        for _ in range(n_car):
            y = rng.choice([2008, 2009, 2010])
            m = rng.randint(1, 13)
            d = rng.randint(1, 28)
            from_dt = f"{y}{m:02d}{d:02d}"
            line_pmts = [round(float(np.exp(rng.normal(4.2, 0.6))), 2) for _ in range(3)]

            rec = {
                "DESYNPUF_ID": m_id,
                "CLM_ID": f"CAR_CLM_{car_counter:07d}",
                "CLM_FROM_DT": from_dt,
                "LINE_NCH_PMT_AMT_1": line_pmts[0],
                "LINE_NCH_PMT_AMT_2": line_pmts[1],
                "LINE_NCH_PMT_AMT_3": line_pmts[2],
            }
            car_records.append(rec)
            car_counter += 1

    half = len(car_records) // 2
    car1_df = pd.DataFrame(car_records[:half])
    car2_df = pd.DataFrame(car_records[half:])

    car1_df.to_csv(raw_dir / "DE1_0_2008_to_2010_Carrier_Claims_Sample_1A.csv", index=False)
    car2_df.to_csv(raw_dir / "DE1_0_2008_to_2010_Carrier_Claims_Sample_1B.csv", index=False)

    return raw_dir


if __name__ == "__main__":
    generate_synthetic_synpuf()
    print("Generated synthetic CMS SynPUF Sample 1 in data/raw/synpuf/")
