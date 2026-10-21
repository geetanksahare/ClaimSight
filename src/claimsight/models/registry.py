import json
import subprocess
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import sklearn

from claimsight.config import get_settings
from claimsight.logging_config import get_logger

logger = get_logger(__name__)


def _get_git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        return out
    except Exception:
        return "unknown"


def save_model(module: str, model: Any, metadata: dict[str, Any]) -> Path:
    """Serialize model artifact and metadata to versioned directory."""
    cfg = get_settings()
    now = datetime.now(timezone.utc)
    version = metadata.get("version", now.strftime("%Y%m%d_%H%M%S"))

    module_dir = cfg.paths.models_dir / module
    version_dir = module_dir / version
    version_dir.mkdir(parents=True, exist_ok=True)

    model_path = version_dir / "model.joblib"
    joblib.dump(model, model_path)

    full_metadata = {
        "module": module,
        "version": version,
        "saved_at_utc": now.isoformat(),
        "git_commit": _get_git_commit(),
        "sklearn_version": sklearn.__version__,
        **metadata,
    }

    metadata_path = version_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(full_metadata, f, indent=2, default=str)

    # Update latest pointer (JSON instead of symlinks for Windows compatibility)
    latest_path = module_dir / "latest.json"
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump({"latest_version": version, "path": str(version_dir)}, f, indent=2)

    logger.info("Saved %s model version %s to %s", module, version, version_dir)
    return version_dir


def load_model(module: str, version: str = "latest") -> tuple[Any, dict[str, Any]]:
    """Load model artifact and metadata, verifying scikit-learn version compatibility."""
    cfg = get_settings()
    module_dir = cfg.paths.models_dir / module

    if not module_dir.exists():
        raise FileNotFoundError(
            f"No model directory found for '{module}'. Run 'claimsight train-{module}' to train."
        )

    if version == "latest":
        latest_file = module_dir / "latest.json"
        if not latest_file.exists():
            raise FileNotFoundError(
                f"No latest.json pointer found for '{module}'. Run 'claimsight train-{module}' first."
            )
        with open(latest_file, "r", encoding="utf-8") as f:
            pointer = json.load(f)
        resolved_version = pointer["latest_version"]
    else:
        resolved_version = version

    version_dir = module_dir / resolved_version
    model_path = version_dir / "model.joblib"
    metadata_path = version_dir / "metadata.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Model file missing at {model_path}")

    model = joblib.load(model_path)

    metadata: dict[str, Any] = {}
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        saved_sklearn = metadata.get("sklearn_version", "")
        if saved_sklearn:
            saved_major_minor = ".".join(saved_sklearn.split(".")[:2])
            current_major_minor = ".".join(sklearn.__version__.split(".")[:2])
            if saved_major_minor != current_major_minor:
                msg = (
                    f"Model '{module}' was trained with scikit-learn {saved_sklearn}, "
                    f"but runtime is {sklearn.__version__}."
                )
                warnings.warn(msg, UserWarning, stacklevel=2)
                logger.warning(msg)

    return model, metadata
