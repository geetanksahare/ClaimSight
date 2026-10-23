from typer.testing import CliRunner

from claimsight.cli import app

runner = CliRunner()


def test_cli_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "ClaimSight command-line interface" in result.output
    assert "train-cost" in result.output
    assert "train-member" in result.output
    assert "train-fraud" in result.output
    assert "info" in result.output


def test_cli_info():
    result = runner.invoke(app, ["info"])
    assert result.exit_code == 0
    assert "ClaimSight Health Insurance Intelligence Platform" in result.output
    assert "Seed: 42" in result.output


def test_cli_train_cost_invokes_pipeline(monkeypatch):
    called = {}

    def mock_train(data_path=None, optuna_trials=None):
        called["data_path"] = data_path
        called["optuna_trials"] = optuna_trials
        return {"test_metrics": {"r2": 0.8888, "rmse": 3200.0}}

    import claimsight.models.train_cost as tc

    monkeypatch.setattr(tc, "train_cost_pipeline", mock_train)

    result = runner.invoke(app, ["train-cost", "--trials", "5"])
    assert result.exit_code == 0
    assert "Finished train-cost" in result.output
    assert called["optuna_trials"] == 5


def test_cli_train_member_invokes_pipeline(monkeypatch):
    called = {}

    def mock_train(trials=None):
        called["trials"] = trials
        return {"metrics": {"pr_auc": 0.25, "test_r2": 0.05, "test_rmse": 2400.0}}

    import claimsight.models.train_member as tm

    monkeypatch.setattr(tm, "train_member_pipeline", mock_train)

    result = runner.invoke(app, ["train-member", "--trials", "3"])
    assert result.exit_code == 0
    assert "Finished train-member" in result.output


def test_cli_train_fraud_invokes_pipeline(monkeypatch):
    called = {}

    def mock_train(trials=None):
        called["trials"] = trials
        return {
            "test_metrics": {"pr_auc": 0.95, "positive_base_rate": 0.1, "roc_auc": 0.99, "f1": 0.92}
        }

    import claimsight.models.train_fraud as tf

    monkeypatch.setattr(tf, "train_fraud_pipeline", mock_train)

    result = runner.invoke(app, ["train-fraud", "--trials", "2"])
    assert result.exit_code == 0
    assert "Finished train-fraud" in result.output


def test_cli_load_and_clean_invokes_pipeline(monkeypatch):
    cleaned = []

    def mock_clean(d):
        cleaned.append(d)

    import claimsight.data.load_and_clean as lc

    monkeypatch.setattr(lc, "load_and_clean", mock_clean)

    result = runner.invoke(app, ["load-and-clean", "--dataset", "cost"])
    assert result.exit_code == 0
    assert "Cleaning dataset: cost" in result.output
    assert cleaned == ["cost"]
