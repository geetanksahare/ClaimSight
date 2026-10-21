from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from claimsight.api.schemas import MemberInput
from claimsight.features.member_features import CHRONIC_CONDITIONS
from claimsight.models.registry import load_model


@lru_cache(maxsize=1)
def _get_member_bundle() -> tuple[dict[str, Any], dict[str, Any]]:
    return load_model("member")


def _assign_tier(cost: float, cutoffs: list[float]) -> str:
    if len(cutoffs) < 3:
        cutoffs = [5000.0, 15000.0, 35000.0]
    if cost < cutoffs[0]:
        return "Low"
    elif cost < cutoffs[1]:
        return "Moderate"
    elif cost < cutoffs[2]:
        return "High"
    return "Very High"


def _member_input_to_df(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for r in records:
        row: dict[str, Any] = {
            "age_at_year_end": r["age"],
            "sex": r["sex"],
            "race": str(r.get("race", "1")),
            "esrd": int(r.get("esrd", 0)),
            "prior_cost_ip": float(r.get("prior_cost_ip", 0.0)),
            "prior_cost_op": float(r.get("prior_cost_op", 0.0)),
            "prior_cost_car": float(r.get("prior_cost_car", 0.0)),
            "ip_claims": int(r.get("ip_claims", 0)),
            "op_claims": int(r.get("op_claims", 0)),
            "carrier_claims": int(r.get("carrier_claims", 0)),
            "ip_days": int(r.get("ip_days", 0)),
            "has_history": int(r.get("has_history", 0)),
            "cost_change": float(r.get("cost_change", 0.0)),
        }

        # Chronic flags unpack
        c_flags = r.get("chronic_flags", {})
        chronic_sum = 0
        for cond in CHRONIC_CONDITIONS:
            val = int(c_flags.get(cond, r.get(cond, 0)))
            row[cond] = val
            chronic_sum += val

        row["chronic_count"] = chronic_sum
        tot_claims = row["ip_claims"] + row["op_claims"] + row["carrier_claims"]
        tot_cost = row["prior_cost_ip"] + row["prior_cost_op"] + row["prior_cost_car"]
        row["prior_total_cost"] = tot_cost
        row["prior_total_claims"] = tot_claims
        row["cost_per_claim"] = tot_cost / max(tot_claims, 1)

        rows.append(row)
    return pd.DataFrame(rows)


def score_member_risk(
    member: dict[str, Any] | pd.DataFrame,
) -> dict[str, Any] | pd.DataFrame:
    """Predict annual expected medical cost, high-cost probability, and risk tier."""
    bundle, metadata = _get_member_bundle()
    regressor = bundle["regressor"]
    classifier = bundle["classifier"]
    cutoffs = metadata.get("tier_cutoffs", [5000.0, 15000.0, 35000.0])

    is_dict = isinstance(member, dict)
    if is_dict:
        validated = MemberInput(**member).model_dump()
        df_in = _member_input_to_df([validated])
    else:
        records = member.to_dict(orient="records")
        validated_records = [MemberInput(**r).model_dump() for r in records]
        df_in = _member_input_to_df(validated_records)

    # Features order matching training
    expected_costs = regressor.predict(df_in)
    probs_high_cost = classifier.predict_proba(df_in)[:, 1]

    tiers = [_assign_tier(float(c), cutoffs) for c in expected_costs]

    # Top drivers from local contribution analysis
    from claimsight.explain.explain_prediction import explain_prediction

    if is_dict:
        exp = explain_prediction(
            "member", df_in.iloc[0:1], top_k=3, model_bundle=(bundle, metadata)
        )
        top_drivers = [c["text"] for c in exp["contributions"]]

        return {
            "expected_cost": round(float(expected_costs[0]), 2),
            "prob_high_cost": round(float(probs_high_cost[0]), 4),
            "risk_tier": tiers[0],
            "top_drivers": top_drivers,
        }

    out_df = member.copy()
    out_df["expected_cost"] = np.round(expected_costs, 2)
    out_df["prob_high_cost"] = np.round(probs_high_cost, 4)
    out_df["risk_tier"] = tiers
    return out_df
