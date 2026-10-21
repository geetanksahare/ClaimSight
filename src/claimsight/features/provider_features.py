import pandas as pd

from claimsight.config import Settings


def build_provider_features(data: dict[str, pd.DataFrame], cfg: Settings) -> pd.DataFrame:
    """Stub for provider fraud detection features."""
    raise NotImplementedError("Provider features implemented in Phase 4")


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate provider features and fraud labels."""
    if "is_fraud" not in df.columns:
        raise ValueError("Target column 'is_fraud' missing from DataFrame")
    X = df.drop(columns=["is_fraud"])
    y = df["is_fraud"]
    return X, y
