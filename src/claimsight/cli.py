from pathlib import Path

import typer

from claimsight.logging_config import configure_logging, get_logger
from claimsight.reporting.model_card import generate_model_card

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
def train_fraud_command() -> None:
    """Train provider anomaly and fraud risk scoring pipeline."""
    configure_logging()
    typer.echo("Executing train-fraud command...")


@app.command("generate-cards")
def generate_cards_command(
    module: str = typer.Option("all", help="Module name (cost, member, fraud, or all)"),
) -> None:
    """Generate model cards for trained modules."""
    configure_logging()
    modules = ["cost", "member", "fraud"] if module == "all" else [module]
    for m in modules:
        try:
            card_path = generate_model_card(m)
            typer.echo(f"Generated card for {m}: {card_path}")
        except Exception as e:
            logger.warning("Could not generate card for %s: %s", m, e)


if __name__ == "__main__":
    app()
