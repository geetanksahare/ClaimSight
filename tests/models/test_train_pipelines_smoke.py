import pytest

from claimsight.config import get_settings
from claimsight.models.train_cost import train_cost_pipeline
from claimsight.models.train_fraud import train_fraud_pipeline
from claimsight.models.train_member import train_member_pipeline


@pytest.fixture
def isolated_dirs(tmp_path, monkeypatch):
    import claimsight.config as cfg_mod

    cfg_mod.get_settings.cache_clear()
    orig_cfg = cfg_mod.load_settings()
    isolated_cfg = orig_cfg.model_copy(
        update={
            "paths": orig_cfg.paths.model_copy(
                update={
                    "models_dir": tmp_path / "models",
                    "reports_dir": tmp_path / "reports",
                }
            )
        }
    )
    monkeypatch.setattr(cfg_mod, "get_settings", lambda: isolated_cfg)
    for mod in [
        "claimsight.models.train_cost",
        "claimsight.models.train_member",
        "claimsight.models.train_fraud",
        "claimsight.models.registry",
        "claimsight.reporting.model_card",
        "claimsight.explain.explain_prediction",
        "claimsight.evaluation.fairness_audit",
    ]:
        monkeypatch.setattr(f"{mod}.get_settings", lambda: isolated_cfg)

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
