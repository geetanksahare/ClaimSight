import pandas as pd

from claimsight.config import Settings


def build_member_features(data: dict[str, pd.DataFrame], cfg: Settings) -> pd.DataFrame:
    """Stub for member longitudinal features."""
    raise NotImplementedError("Member features implemented in Phase 3")


def label_high_cost(y_train: pd.Series, y_all: pd.Series, q: float) -> tuple[float, pd.Series]:
    """Calculate high-cost cutoff on training target and apply to full series."""
    threshold = float(y_train.quantile(q))
    labels = (y_all >= threshold).astype(int)
    return threshold, labels
