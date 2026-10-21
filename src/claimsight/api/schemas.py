from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class CostInput(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "age": 35,
                "sex": "male",
                "bmi": 28.5,
                "children": 2,
                "smoker": "no",
                "region": "northwest",
            }
        },
    )

    age: int = Field(..., ge=18, le=100, description="Age in years (18 to 100)")
    sex: Literal["male", "female"] = Field(..., description="Biological sex")
    bmi: float = Field(..., ge=10.0, le=70.0, description="Body Mass Index (10.0 to 70.0)")
    children: int = Field(..., ge=0, le=10, description="Number of dependent children (0 to 10)")
    smoker: Literal["yes", "no"] = Field(..., description="Smoking status")
    region: Literal["northeast", "northwest", "southeast", "southwest"] = Field(
        ..., description="US geographic region"
    )


class MemberInput(BaseModel):
    model_config = ConfigDict(
        extra="ignore",
        json_schema_extra={
            "example": {
                "member_id": "MEM_001",
                "age": 68,
                "sex": "female",
                "race": "1",
                "esrd": 0,
                "chronic_flags": {
                    "SP_ALZHDMTA": 0,
                    "SP_CHF": 1,
                    "SP_CHRNKIDN": 0,
                    "SP_CNCR": 0,
                    "SP_COPD": 1,
                    "SP_DEPRESSN": 0,
                    "SP_DIABETES": 1,
                    "SP_ISCHMCHT": 1,
                    "SP_OSTEOPRS": 0,
                    "SP_RA_OA": 0,
                    "SP_STRKETIA": 0,
                },
                "prior_cost_ip": 1200.0,
                "prior_cost_op": 850.0,
                "prior_cost_car": 640.0,
                "ip_claims": 1,
                "op_claims": 4,
                "carrier_claims": 6,
                "ip_days": 3,
                "has_history": 1,
                "cost_change": 450.0,
            }
        },
    )

    member_id: str = Field(default="MEMBER_001")
    age: int = Field(..., ge=0, le=120)
    sex: Literal["male", "female"]
    race: str = Field(default="1")
    esrd: int = Field(default=0, ge=0, le=1)
    chronic_flags: dict[str, int] = Field(default_factory=dict)
    prior_cost_ip: float = Field(default=0.0, ge=0.0)
    prior_cost_op: float = Field(default=0.0, ge=0.0)
    prior_cost_car: float = Field(default=0.0, ge=0.0)
    ip_claims: int = Field(default=0, ge=0)
    op_claims: int = Field(default=0, ge=0)
    carrier_claims: int = Field(default=0, ge=0)
    ip_days: int = Field(default=0, ge=0)
    has_history: int = Field(default=0, ge=0, le=1)
    cost_change: float = Field(default=0.0)


class ProviderFeatures(BaseModel):
    model_config = ConfigDict(extra="ignore")

    Provider: str = Field(..., description="Provider identifier")
    n_claims: int = Field(..., ge=1)
    n_inpatient: int = Field(default=0, ge=0)
    n_outpatient: int = Field(default=0, ge=0)
    inpatient_share: float = Field(default=0.0, ge=0.0, le=1.0)
    n_unique_beneficiaries: int = Field(default=1, ge=1)
    claims_per_beneficiary: float = Field(default=1.0, ge=0.0)
    reimb_sum: float = Field(default=0.0, ge=0.0)
    reimb_mean: float = Field(default=0.0, ge=0.0)
    reimb_median: float = Field(default=0.0, ge=0.0)
    reimb_max: float = Field(default=0.0, ge=0.0)
    reimb_std: float = Field(default=0.0, ge=0.0)
    deductible_mean: float = Field(default=0.0, ge=0.0)
    n_unique_attending: int = Field(default=1, ge=0)
    n_unique_operating: int = Field(default=0, ge=0)
    attending_missing_share: float = Field(default=0.0, ge=0.0, le=1.0)
    operating_missing_share: float = Field(default=0.0, ge=0.0, le=1.0)
    top_attending_claim_share: float = Field(default=0.0, ge=0.0, le=1.0)
    mean_los: float = Field(default=0.0, ge=0.0)
    mean_claim_duration: float = Field(default=0.0, ge=0.0)
    mean_n_diagnosis_codes: float = Field(default=0.0, ge=0.0)
    mean_n_procedure_codes: float = Field(default=0.0, ge=0.0)
    n_unique_diag_codes: int = Field(default=0, ge=0)
    bene_mean_age: float = Field(default=65.0, ge=0.0)
    bene_chronic_mean: float = Field(default=0.0, ge=0.0)
    bene_renal_share: float = Field(default=0.0, ge=0.0, le=1.0)
    bene_mean_partA_months: float = Field(default=12.0, ge=0.0)
    share_claims_same_day_multiple: float = Field(default=0.0, ge=0.0, le=1.0)
    is_fraud: int | None = Field(default=None)


class FlagRequest(BaseModel):
    providers: list[ProviderFeatures]
    top_n: int | None = Field(default=None, ge=1)


class WhatIfRequest(BaseModel):
    base: CostInput
    changes: dict[str, Any]
