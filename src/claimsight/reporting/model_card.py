import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader

from claimsight.config import get_settings
from claimsight.logging_config import get_logger
from claimsight.models.registry import load_model

logger = get_logger(__name__)


def generate_model_card(module: str) -> Path:
    """Render and save a comprehensive markdown model card from evaluated metrics and metadata."""
    cfg = get_settings()
    model, metadata = load_model(module)

    metrics_file = cfg.paths.reports_dir / "metrics" / f"{module}.json"
    metrics_data: dict[str, Any] = {}
    if metrics_file.exists():
        with open(metrics_file, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)

    fairness_file = cfg.paths.reports_dir / "metrics" / f"fairness_{module}.json"
    fairness_summary: dict[str, Any] = {}
    if fairness_file.exists():
        with open(fairness_file, "r", encoding="utf-8") as f:
            f_data = json.load(f)
            fairness_summary = f_data.get("columns", {})

    template_dir = Path(__file__).parent / "templates"
    env = Environment(loader=FileSystemLoader(str(template_dir)), autoescape=False)
    template = env.get_template("model_card.md.j2")

    coverage_val = round(metrics_data.get("conformal_coverage", 0.0) * 100, 1)

    template_context = {
        "module": module,
        "module_name": f"{module.title()} Intelligence Model",
        "version": metadata.get("version", "latest"),
        "model_type": metadata.get("model_type", type(model).__name__),
        "generated_date": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "git_commit": metadata.get("git_commit", "unknown"),
        "sklearn_version": metadata.get("sklearn_version", "unknown"),
        "data_source": metadata.get("data_source", "Public Kaggle / CMS Public Use File"),
        "training_rows": metadata.get("training_rows", "N/A"),
        "features": metadata.get("features", []),
        "split_strategy": metadata.get("split_strategy", "Stratified cross-validation"),
        "seed": metadata.get("seed", cfg.seed),
        "metrics": metrics_data,
        "conformal_q": metadata.get("conformal_q"),
        "interval_coverage": coverage_val,
        "fairness_summary": fairness_summary,
        "train_command": f"train-{module}",
    }

    rendered_card = template.render(**template_context)

    out_path = cfg.paths.reports_dir / "model_cards" / f"{module}.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(rendered_card)

    logger.info("Generated model card at %s", out_path)
    return out_path
