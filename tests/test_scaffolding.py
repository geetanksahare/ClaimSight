from claimsight.config import get_settings, load_settings
from claimsight.logging_config import configure_logging, get_logger


def test_settings_load():
    settings = get_settings()
    assert settings.seed == 42
    assert settings.cost.test_size == 0.15
    assert settings.member.high_cost_quantile == 0.90
    assert settings.fraud.risk_weights.supervised == 0.7
    assert settings.paths.raw_dir.name == "raw"


def test_logging_configuration():
    configure_logging()
    logger = get_logger("claimsight.test")
    assert logger is not None
    assert logger.name == "claimsight.test"


def test_load_settings_explicit(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
seed: 123
paths:
  raw_dir: "data/raw"
  interim_dir: "data/interim"
  processed_dir: "data/processed"
  models_dir: "models"
  reports_dir: "reports"
cost:
  test_size: 0.2
  calib_size: 0.1
  cv_folds: 3
  optuna_trials: 5
  interval_alpha: 0.1
  bmi_bins: [0, 25, 100]
  bmi_labels: ["normal", "high"]
  age_bins: [18, 50, 100]
  age_labels: ["young", "old"]
member:
  high_cost_quantile: 0.85
  tier_quantiles: [0.5, 0.8, 0.95]
  tweedie_power: 1.5
  cv_folds: 3
  optuna_trials: 5
fraud:
  test_size: 0.2
  cv_folds: 3
  optuna_trials: 5
  threshold_metric: "f1"
  iforest_contamination: 0.05
  risk_weights:
    supervised: 0.6
    anomaly: 0.4
fairness:
  relative_gap_threshold: 0.15
  sensitive_cost: ["sex"]
  sensitive_member: ["race"]
api:
  max_batch_rows: 1000
  cors_origins: ["*"]
"""
    )
    settings = load_settings(config_file)
    assert settings.seed == 123
    assert settings.cost.cv_folds == 3
