from pathlib import Path

import typer

from claimsight.logging_config import configure_logging, get_logger

app = typer.Typer(help="ClaimSight command-line interface for models, cards, and evaluation.")
logger = get_logger(__name__)


@app.command("train-cost")
def train_cost_command(
    data_path: Path | None = typer.Option(None, help="Custom path to cost data"),
    trials: int | None = typer.Option(None, help="Optuna hyperparameter tuning trials"),
) -> None:
    """Train cost prediction pipeline, conformal calibration, and model card."""
    configure_logging()
    from claimsight.models.train_cost import train_cost_pipeline

    typer.echo("Executing train-cost command...")
    results = train_cost_pipeline(data_path=data_path, optuna_trials=trials)
    metrics = results["test_metrics"]
    typer.echo(
        f"Finished train-cost: Test R2 = {metrics['r2']:.4f}, Test RMSE = ${metrics['rmse']:,.2f}"
    )


@app.command("train-member")
def train_member_command(
    trials: int | None = typer.Option(None, help="Optuna hyperparameter tuning trials"),
) -> None:
    """Train member risk regression and classification pipeline."""
    configure_logging()
    from claimsight.models.train_member import train_member_pipeline

    typer.echo("Executing train-member command...")
    results = train_member_pipeline(trials=trials)
    metrics = results["metrics"]
    typer.echo(
        f"Finished train-member: PR-AUC (High Cost) = {metrics['pr_auc']:.4f}, "
        f"Expected Cost Test R2 = {metrics['test_r2']:.4f}, RMSE = ${metrics['test_rmse']:,.2f}"
    )


@app.command("train-fraud")
def train_fraud_command(
    trials: int | None = typer.Option(None, help="Optuna hyperparameter tuning trials"),
) -> None:
    """Train provider anomaly and fraud risk scoring pipeline."""
    configure_logging()
    from claimsight.models.train_fraud import train_fraud_pipeline

    typer.echo("Executing train-fraud command...")
    results = train_fraud_pipeline(trials=trials)
    metrics = results["test_metrics"]
    typer.echo(
        f"Finished train-fraud: PR-AUC = {metrics['pr_auc']:.4f} (Base rate: {metrics['positive_base_rate']:.4f}), "
        f"ROC-AUC = {metrics['roc_auc']:.4f}, F1 = {metrics['f1']:.4f}"
    )


@app.command("load-and-clean")
def load_and_clean_command(
    dataset: str = typer.Option("all", help="Dataset to clean (cost, synpuf, fraud, or all)"),
) -> None:
    """Download or clean raw datasets and write processed parquet files."""
    configure_logging()
    from claimsight.data.load_and_clean import load_and_clean

    targets = ["cost", "synpuf", "fraud"] if dataset == "all" else [dataset]
    for d in targets:
        typer.echo(f"Cleaning dataset: {d}...")
        load_and_clean(d)
    typer.echo("Data processing completed.")


@app.command("info")
def info_command() -> None:
    """Display environment, configuration, and registered model details."""
    from claimsight.config import get_settings
    from claimsight.models.registry import load_model

    cfg = get_settings()
    typer.echo("ClaimSight Health Insurance Intelligence Platform")
    typer.echo(f"Seed: {cfg.seed}")
    typer.echo(f"Environment: {cfg.env}")
    for m in ["cost", "member", "fraud"]:
        try:
            _, meta = load_model(m)
            typer.echo(
                f"Model {m}: Version {meta.get('version')} (Trained: {meta.get('created_at')})"
            )
        except Exception:
            typer.echo(f"Model {m}: Not trained")


if __name__ == "__main__":
    app()
