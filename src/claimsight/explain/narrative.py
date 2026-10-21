from typing import Any

FEATURE_LABELS: dict[str, str] = {
    # Cost features
    "age": "age",
    "bmi": "body mass index (BMI)",
    "children": "number of dependent children",
    "sex": "biological sex",
    "smoker": "smoking status",
    "region": "geographic region",
    "age_band": "age band",
    "bmi_category": "BMI category",
    "is_obese": "obesity indicator",
    "smoker_flag": "smoking status",
    "sex_flag": "sex indicator",
    "has_children": "parental status",
    "smoker_x_obese": "smoking combined with obesity",
    "smoker_x_age": "smoking combined with age",
    "bmi_x_smoker": "BMI interaction with smoking",
    "family_size_bucket": "family size category",
    # Member features
    "age_at_year_end": "age at year end",
    "chronic_count": "number of chronic conditions",
    "prior_total_cost": "prior year total medical cost",
    "prior_total_claims": "prior year claim frequency",
    "cost_per_claim": "prior cost per claim",
    "cost_change": "year-over-year cost change",
    "ip_days": "inpatient utilization days",
    "ip_pmt": "inpatient reimbursement",
    "op_pmt": "outpatient reimbursement",
    "carrier_pmt": "carrier reimbursement",
    # Fraud / provider features
    "n_claims": "number of claims billed",
    "reimb_sum": "total reimbursed amount",
    "reimb_mean": "average claim reimbursement",
    "reimb_max": "maximum single claim reimbursement",
    "claims_per_beneficiary": "claims per unique beneficiary",
    "top_attending_claim_share": "physician billing concentration",
    "share_claims_same_day_multiple": "repeat same-day billing share",
    "mean_n_procedure_codes": "mean procedure codes per claim",
    "mean_claim_duration": "average claim duration",
    "inpatient_share": "inpatient share of claims",
}


def to_text(
    feature: str,
    value: Any,
    shap_val: float,
    effect_pct: float | None = None,
    module: str = "cost",
) -> str:
    """Generate neutral, non-causal plain language explanation for a feature contribution."""
    label = FEATURE_LABELS.get(feature, feature.replace("_", " "))
    val_str = f" ({value})" if value is not None and str(value) != "" else ""

    if module == "cost":
        if effect_pct is not None:
            pct_abs = abs(effect_pct)
            direction = "raises" if effect_pct >= 0 else "lowers"
            return (
                f"{label.capitalize()}{val_str} is associated with a change that "
                f"{direction} the estimate by about {pct_abs:.1f}% versus the baseline."
            )
        direction = "increases" if shap_val >= 0 else "decreases"
        return f"{label.capitalize()}{val_str} {direction} the expected cost versus the baseline."

    # Member or fraud modules
    direction = "raises" if shap_val >= 0 else "lowers"
    score_name = "risk score" if module == "fraud" else "expected risk tier"
    return f"{label.capitalize()}{val_str} is associated with a factor that {direction} the {score_name} versus the baseline."
