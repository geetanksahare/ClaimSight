import pandera.pandas as pa
from pandera.pandas import Check, Column, DataFrameSchema

cost_clean_schema = DataFrameSchema(
    {
        "row_id": Column(pa.Int64, Check.ge(0), nullable=False),
        "age": Column(pa.Int64, Check.in_range(18, 64), nullable=False),
        "sex": Column(pa.String, Check.isin(["male", "female"]), nullable=False),
        "bmi": Column(pa.Float64, Check.in_range(10.0, 60.0), nullable=False),
        "children": Column(pa.Int64, Check.in_range(0, 5), nullable=False),
        "smoker": Column(pa.String, Check.isin(["yes", "no"]), nullable=False),
        "region": Column(
            pa.String,
            Check.isin(["northeast", "northwest", "southeast", "southwest"]),
            nullable=False,
        ),
        "charges": Column(pa.Float64, Check.gt(0.0), nullable=False),
    },
    strict=True,
    coerce=True,
)

synpuf_beneficiary_schema = DataFrameSchema(
    {
        "DESYNPUF_ID": Column(pa.String, nullable=False),
        "year": Column(pa.Int64, Check.isin([2008, 2009, 2010]), nullable=False),
        "BENE_BIRTH_DT": Column(pa.DateTime, nullable=True),
        "BENE_DEATH_DT": Column(pa.DateTime, nullable=True),
        "BENE_SEX_IDENT_CD": Column(pa.String, Check.isin(["male", "female"]), nullable=False),
        "BENE_RACE_CD": Column(pa.String, nullable=False),
        "BENE_ESRD_IND": Column(pa.Int64, Check.isin([0, 1]), nullable=False),
        "SP_STATE_CODE": Column(pa.String, nullable=False),
        "BENE_HI_CVRAGE_TOT_MONS": Column(pa.Int64, Check.in_range(0, 12), nullable=False),
        "BENE_SMI_CVRAGE_TOT_MONS": Column(pa.Int64, Check.in_range(0, 12), nullable=False),
        "MEDREIMB_IP": Column(pa.Float64, Check.ge(0.0), nullable=False),
        "MEDREIMB_OP": Column(pa.Float64, Check.ge(0.0), nullable=False),
        "MEDREIMB_CAR": Column(pa.Float64, Check.ge(0.0), nullable=False),
    },
    strict=False,
    coerce=True,
)

synpuf_inpatient_schema = DataFrameSchema(
    {
        "DESYNPUF_ID": Column(pa.String, nullable=False),
        "CLM_ID": Column(pa.String, nullable=False),
        "CLM_FROM_DT": Column(pa.DateTime, nullable=True),
        "CLM_THRU_DT": Column(pa.DateTime, nullable=True),
        "CLM_PMT_AMT": Column(pa.Float64, nullable=False),
        "CLM_UTLZTN_DAY_CNT": Column(pa.Float64, Check.ge(0.0), nullable=False),
        "year": Column(pa.Int64, nullable=False),
    },
    strict=False,
    coerce=True,
)

synpuf_outpatient_schema = DataFrameSchema(
    {
        "DESYNPUF_ID": Column(pa.String, nullable=False),
        "CLM_ID": Column(pa.String, nullable=False),
        "CLM_FROM_DT": Column(pa.DateTime, nullable=True),
        "CLM_PMT_AMT": Column(pa.Float64, nullable=False),
        "year": Column(pa.Int64, nullable=False),
    },
    strict=False,
    coerce=True,
)

synpuf_carrier_agg_schema = DataFrameSchema(
    {
        "DESYNPUF_ID": Column(pa.String, nullable=False),
        "year": Column(pa.Int64, nullable=False),
        "carrier_claims": Column(pa.Int64, Check.ge(0), nullable=False),
        "carrier_pmt": Column(pa.Float64, Check.ge(0.0), nullable=False),
    },
    strict=True,
    coerce=True,
)

fraud_labels_schema = DataFrameSchema(
    {
        "Provider": Column(pa.String, nullable=False),
        "is_fraud": Column(pa.Int64, Check.isin([0, 1]), nullable=False),
    },
    strict=True,
    coerce=True,
)
