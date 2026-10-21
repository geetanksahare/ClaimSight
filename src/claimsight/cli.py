import typer

app = typer.Typer(help="ClaimSight command-line interface for models, cards, and evaluation.")


@app.command("train-cost")
def train_cost_command() -> None:
    """Train cost prediction pipeline, conformal calibration, and model card."""
    typer.echo("Executing train-cost command...")


@app.command("train-member")
def train_member_command() -> None:
    """Train member risk regression and classification pipeline."""
    typer.echo("Executing train-member command...")


@app.command("train-fraud")
def train_fraud_command() -> None:
    """Train provider anomaly and fraud risk scoring pipeline."""
    typer.echo("Executing train-fraud command...")


@app.command("generate-cards")
def generate_cards_command() -> None:
    """Generate model cards for all trained modules."""
    typer.echo("Generating model cards...")


if __name__ == "__main__":
    app()
