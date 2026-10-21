import json
from typing import Any, Literal

import numpy as np
import pandas as pd
from fairlearn.metrics import MetricFrame
from sklearn.metrics import (
    mean_absolute_error,
    precision_score,
    root_mean_squared_error,
)

from claimsight.config import get_settings


def mean_signed_error(y_true: Any, y_pred: Any) -> float:
    return float(np.mean(np.asarray(y_pred) - np.asarray(y_true)))


def mean_prediction(y_true: Any, y_pred: Any) -> float:
    return float(np.mean(np.asarray(y_pred)))


def selection_rate(y_true: Any, y_pred: Any) -> float:
    return float(np.mean(np.asarray(y_pred)))


def true_positive_rate(y_true: Any, y_pred: Any) -> float:
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)
    positives = y_t == 1
    if not np.any(positives):
        return 0.0
    return float(np.mean(y_p[positives]))


def false_positive_rate(y_true: Any, y_pred: Any) -> float:
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)
    negatives = y_t == 0
    if not np.any(negatives):
        return 0.0
    return float(np.mean(y_p[negatives]))


def fairness_audit(
    y_true: Any,
    y_pred: Any,
    sensitive: pd.DataFrame,
    task: Literal["regression", "classification"],
    y_prob: Any = None,
    threshold: float | None = None,
    relative_gap_threshold: float = 0.20,
    module: str = "cost",
) -> dict[str, Any]:
    """Audit subgroup disparities using Fairlearn MetricFrame and generate disparity report."""
    cfg = get_settings()
    reports_dir = cfg.paths.reports_dir

    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)

    if task == "classification":
        thresh = 0.5 if threshold is None else float(threshold)
        if y_prob is not None:
            probs = np.asarray(y_prob)
            y_pred_binary = (probs >= thresh).astype(int)
        else:
            y_pred_binary = (y_p >= thresh).astype(int)

        metrics_dict = {
            "selection_rate": selection_rate,
            "tpr": true_positive_rate,
            "fpr": false_positive_rate,
            "precision": lambda yt, yp: float(precision_score(yt, yp, zero_division=0)),
            "count": lambda yt, yp: int(len(yt)),
        }
        eval_pred = y_pred_binary
        primary_eval = "selection_rate"
    else:
        metrics_dict = {
            "rmse": root_mean_squared_error,
            "mae": mean_absolute_error,
            "mean_signed_error": mean_signed_error,
            "mean_prediction": mean_prediction,
            "count": lambda yt, yp: int(len(yt)),
        }
        eval_pred = y_p
        primary_eval = "mae"

    audit_results: dict[str, Any] = {
        "task": task,
        "module": module,
        "relative_gap_threshold": relative_gap_threshold,
        "columns": {},
    }

    for col in sensitive.columns:
        sens_series = sensitive[col].astype(str)
        mf = MetricFrame(
            metrics=metrics_dict,
            y_true=y_t,
            y_pred=eval_pred,
            sensitive_features=sens_series,
        )

        by_group = mf.by_group
        group_records: dict[str, dict[str, Any]] = {}
        valid_metric_vals: list[float] = []

        for grp in by_group.index:
            cnt = int(by_group.loc[grp, "count"])
            low_n = bool(cnt < 30)
            rec = {
                "count": cnt,
                "low_n": low_n,
            }
            for m_name in metrics_dict.keys():
                val = float(by_group.loc[grp, m_name])
                rec[m_name] = round(val, 4)

            group_records[str(grp)] = rec
            if not low_n:
                valid_metric_vals.append(rec[primary_eval])

        overall_val = float(mf.overall[primary_eval])
        if len(valid_metric_vals) >= 2:
            max_val = max(valid_metric_vals)
            min_val = min(valid_metric_vals)
            max_min_gap = max_val - min_val
            rel_gap = max_min_gap / max(abs(overall_val), 1e-6)
        else:
            max_min_gap = 0.0
            rel_gap = 0.0

        disparity_flag = bool(rel_gap > relative_gap_threshold)

        audit_results["columns"][col] = {
            "overall_metric": round(overall_val, 4),
            "primary_metric": primary_eval,
            "max_min_gap": round(max_min_gap, 4),
            "relative_gap": round(rel_gap, 4),
            "disparity_flag": disparity_flag,
            "groups": group_records,
        }

    # Save JSON report
    json_path = reports_dir / "metrics" / f"fairness_{module}.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(audit_results, f, indent=2)

    # Render Markdown summary
    md_path = reports_dir / f"fairness_{module}.md"
    md_lines = [
        f"# Fairness Audit Report: {module.title()} Module",
        "",
        "> This report evaluates demographic subgroup performance parity. It assesses statistical disparities in prediction errors across protected attributes; it neither proves nor disproves systemic discrimination.",
        "",
        f"**Task Type:** {task}  ",
        f"**Disparity Threshold:** {relative_gap_threshold * 100:.0f}% relative gap  ",
        "",
        "## Subgroup Performance Summary",
        "",
    ]

    for col, col_data in audit_results["columns"].items():
        flag_str = "**FLAGGED**" if col_data["disparity_flag"] else "Passed"
        md_lines.append(f"### Attribute: `{col}` ({flag_str})")
        md_lines.append(
            f"- Overall {col_data['primary_metric']}: {col_data['overall_metric']}  \n"
            f"- Max-Min Disparity Gap: {col_data['max_min_gap']} (Relative Gap: {col_data['relative_gap'] * 100:.1f}%)"
        )
        md_lines.append("")
        md_lines.append("| Group | Count | Low N (<30) | Primary Metric |")
        md_lines.append("|---|---|---|---|")
        for grp, g_info in col_data["groups"].items():
            md_lines.append(
                f"| {grp} | {g_info['count']} | {g_info['low_n']} | {g_info[col_data['primary_metric']]} |"
            )
        md_lines.append("")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    return audit_results
