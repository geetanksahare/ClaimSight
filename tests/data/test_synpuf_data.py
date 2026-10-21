from pathlib import Path

import pandas as pd

from claimsight.data.load_and_clean import _clean_synpuf


def test_clean_synpuf_recodes(synpuf_mini_raw_dir: Path):
    cleaned = _clean_synpuf(synpuf_mini_raw_dir)
    bene = cleaned["beneficiary"]

    # Sex recoding: 1 -> male, 2 -> female
    mem1 = bene[bene["DESYNPUF_ID"] == "MEM_01"].iloc[0]
    mem2 = bene[bene["DESYNPUF_ID"] == "MEM_02"].iloc[0]
    assert mem1["BENE_SEX_IDENT_CD"] == "male"
    assert mem2["BENE_SEX_IDENT_CD"] == "female"

    # Chronic conditions recoding: 1 -> 1, 2 -> 0
    assert mem1["SP_ALZHDMTA"] == 1
    assert mem2["SP_ALZHDMTA"] == 0
    assert mem1["SP_CHF"] == 0
    assert mem2["SP_CHF"] == 1

    # ESRD indicator: "0" -> 0, "Y" -> 1
    assert mem1["BENE_ESRD_IND"] == 0
    assert mem2["BENE_ESRD_IND"] == 1

    # Date parsing
    assert pd.to_datetime(mem1["BENE_BIRTH_DT"]).year == 1940
    assert pd.to_datetime(mem2["BENE_BIRTH_DT"]).year == 1935

    # Carrier aggregation
    carrier_agg = cleaned["carrier_agg"]
    assert len(carrier_agg) == 2
    assert set(carrier_agg["DESYNPUF_ID"]) == {"MEM_01", "MEM_02"}
