import pytest

from claimsight.config import get_settings
from claimsight.models.train_cost import train_cost_pipeline
from claimsight.models.train_fraud import train_fraud_pipeline
from claimsight.models.train_member import train_member_pipeline


@pytest.fixture
def isolated_dirs(tmp_path, monkeypatch):
    cfg = get_settings()
    isolated_cfg = cfg.model_copy(
        update={
            "paths": cfg.paths.model_copy(
                update={
                    "models_dir": tmp_path / "models",
                    "reports_dir": tmp_path / "reports",
                }
            )
        }
    )
    monkeypatch.setattr("claimsight.config.get_settings", lambda: isolated_cfg)
    monkeypatch.setattr("claimsight.models.registry.get_settings", lambda: isolated_cfg)
    monkeypatch.setattr("claimsight.models.train_cost.get_settings", lambda: isolated_cfg)
    monkeypatch.setattr("claimsight.models.train_member.get_settings", lambda: isolated_cfg)
    monkeypatch.setattr("claimsight.models.train_fraud.get_settings", lambda: isolated_cfg)
    return isolated_cfg


@pytest.mark.slow
def test_train_cost_pipeline_smoke(isolated_dirs):
    res = train_cost_pipeline(optuna_trials=1)
    assert "test_metrics" in res
    assert res["test_metrics"]["r2"] > 0.80


@pytest.mark.slow
def test_train_fraud_pipeline_smoke(isolated_dirs):
    res = train_fraud_pipeline(trials=1)
    assert "test_metrics" in res
    assert "pr_auc" in res["test_metrics"]


@pytest.mark.slow
def test_train_member_pipeline_smoke(isolated_dirs):
    res = train_member_pipeline(trials=1)
    assert "metrics" in res
    assert "pr_auc" in res["metrics"]
