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
