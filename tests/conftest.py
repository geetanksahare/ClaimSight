from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def synpuf_mini_raw_dir(tmp_path: Path) -> Path:
    """Fixture providing tiny valid SynPUF raw CSV files."""
    raw_dir = tmp_path / "synpuf"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Beneficiary summary 2008
    bene_2008 = pd.DataFrame(
        {
            "DESYNPUF_ID": ["MEM_01", "MEM_02"],
            "BENE_BIRTH_DT": ["19400512", "19351020"],
            "BENE_DEATH_DT": ["", ""],
            "BENE_SEX_IDENT_CD": [1, 2],  # 1 = male, 2 = female
            "BENE_RACE_CD": [1, 2],
            "BENE_ESRD_IND": ["0", "Y"],
            "SP_STATE_CODE": ["01", "05"],
            "BENE_HI_CVRAGE_TOT_MONS": [12, 12],
            "BENE_SMI_CVRAGE_TOT_MONS": [12, 12],
            "SP_ALZHDMTA": [1, 2],  # 1 = yes, 2 = no
            "SP_CHF": [2, 1],
            "MEDREIMB_IP": [0.0, 1500.0],
            "MEDREIMB_OP": [200.0, 500.0],
            "MEDREIMB_CAR": [300.0, 400.0],
        }
    )
    bene_2008.to_csv(raw_dir / "DE1_0_2008_Beneficiary_Summary_File_Sample_1.csv", index=False)

    # Inpatient 2008
    ip_df = pd.DataFrame(
        {
            "DESYNPUF_ID": ["MEM_02"],
            "CLM_ID": ["IP_001"],
            "CLM_FROM_DT": ["20080315"],
            "CLM_THRU_DT": ["20080319"],
            "CLM_PMT_AMT": [1500.0],
            "CLM_UTLZTN_DAY_CNT": [4],
        }
    )
    ip_df.to_csv(raw_dir / "DE1_0_2008_to_2010_Inpatient_Claims_Sample_1.csv", index=False)

    # Outpatient 2008
    op_df = pd.DataFrame(
        {
            "DESYNPUF_ID": ["MEM_01", "MEM_02"],
            "CLM_ID": ["OP_001", "OP_002"],
            "CLM_FROM_DT": ["20080210", "20080614"],
            "CLM_PMT_AMT": [200.0, 500.0],
        }
    )
    op_df.to_csv(raw_dir / "DE1_0_2008_to_2010_Outpatient_Claims_Sample_1.csv", index=False)

    # Carrier 2008
    car_df = pd.DataFrame(
        {
            "DESYNPUF_ID": ["MEM_01", "MEM_02"],
            "CLM_ID": ["CAR_001", "CAR_002"],
            "CLM_FROM_DT": ["20080112", "20080518"],
            "LINE_NCH_PMT_AMT_1": [100.0, 200.0],
            "LINE_NCH_PMT_AMT_2": [200.0, 200.0],
        }
    )
    car_df.to_csv(raw_dir / "DE1_0_2008_to_2010_Carrier_Claims_Sample_1A.csv", index=False)

    return raw_dir
