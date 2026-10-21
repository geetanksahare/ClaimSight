import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

# Ensure MLflow 3.x supports local file store and silence hints
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")


class PathSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    raw_dir: Path
    interim_dir: Path
    processed_dir: Path
    models_dir: Path
    reports_dir: Path


class CostSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    test_size: float = 0.15
    calib_size: float = 0.15
    cv_folds: int = 5
    optuna_trials: int = 50
    interval_alpha: float = 0.10
    bmi_bins: list[float]
    bmi_labels: list[str]
    age_bins: list[int]
    age_labels: list[str]


class MemberSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    high_cost_quantile: float = 0.90
    tier_quantiles: list[float] = Field(default_factory=lambda: [0.50, 0.80, 0.95])
    tweedie_power: float = 1.5
    cv_folds: int = 5
    optuna_trials: int = 40


class FraudRiskWeights(BaseModel):
    model_config = ConfigDict(frozen=True)

    supervised: float = 0.7
    anomaly: float = 0.3


class FraudSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    test_size: float = 0.20
    cv_folds: int = 5
    optuna_trials: int = 40
    threshold_metric: str = "f1"
    iforest_contamination: float = 0.09
    risk_weights: FraudRiskWeights


class FairnessSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    relative_gap_threshold: float = 0.20
    sensitive_cost: list[str]
    sensitive_member: list[str]


class ApiSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    max_batch_rows: int = 50000
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])


class Settings(BaseModel):
    model_config = ConfigDict(frozen=True)

    seed: int = 42
    paths: PathSettings
    cost: CostSettings
    member: MemberSettings
    fraud: FraudSettings
    fairness: FairnessSettings
    api: ApiSettings
    api_key: str | None = None
    mlflow_tracking_uri: str = "./mlruns"
    env: str = "development"


def find_repo_root() -> Path:
    current = Path(__file__).resolve().parent
    for candidate in [current, *current.parents]:
        if (candidate / "configs" / "config.yaml").exists() or (
            candidate / "pyproject.toml"
        ).exists():
            return candidate
    return Path.cwd()


def load_settings(config_path: Path | str | None = None) -> Settings:
    repo_root = find_repo_root()
    if config_path is None:
        resolved_config = repo_root / "configs" / "config.yaml"
    else:
        resolved_config = Path(config_path)
        if not resolved_config.is_absolute():
            resolved_config = (repo_root / resolved_config).resolve()

    if not resolved_config.exists():
        raise FileNotFoundError(f"Configuration file not found: {resolved_config}")

    with open(resolved_config, "r", encoding="utf-8") as f:
        raw_data: dict[str, Any] = yaml.safe_load(f)

    paths_data = raw_data.get("paths", {})
    resolved_paths = {
        key: (repo_root / val).resolve() if not Path(val).is_absolute() else Path(val)
        for key, val in paths_data.items()
    }
    raw_data["paths"] = resolved_paths

    raw_data["api_key"] = os.getenv("CLAIMSIGHT_API_KEY", raw_data.get("api_key"))
    raw_data["mlflow_tracking_uri"] = os.getenv(
        "MLFLOW_TRACKING_URI", raw_data.get("mlflow_tracking_uri", "./mlruns")
    )
    raw_data["env"] = os.getenv("CLAIMSIGHT_ENV", raw_data.get("env", "development"))

    return Settings(**raw_data)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()
