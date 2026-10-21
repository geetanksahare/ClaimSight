from typing import Literal

import pandas as pd

from claimsight.config import Settings
from claimsight.features.cost_features import build_cost_features
from claimsight.features.member_features import build_member_features
from claimsight.features.provider_features import build_provider_features


def build_features(
    dataset: Literal["cost", "member", "fraud"],
    data: pd.DataFrame | dict[str, pd.DataFrame],
    cfg: Settings,
    **kwargs,
) -> pd.DataFrame:
    """Pure feature pipeline dispatcher returning model-ready DataFrame."""
    if dataset == "cost":
        if isinstance(data, dict):
            raise TypeError("Cost dataset requires a single DataFrame")
        return build_cost_features(data, cfg)

    if dataset == "member":
        if not isinstance(data, dict):
            raise TypeError("Member risk pipeline requires dictionary of tables")
        return build_member_features(data, cfg)

    if dataset == "fraud":
        if not isinstance(data, dict):
            raise TypeError("Fraud pipeline requires dictionary of tables")
        return build_provider_features(data, cfg)

    raise ValueError(f"Unknown dataset type: {dataset}")
