import json

import numpy as np

from claimsight.evaluation.evaluate import evaluate, save_metrics
from claimsight.evaluation.plots import (
    plot_calibration_curve,
    plot_error_by_segment,
    plot_interval_coverage,
    plot_pr_curve,
    plot_predicted_vs_actual,
    plot_residuals,
    plot_roc_curve,
)


def test_evaluate_regression():
    y_true = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    y_pred = np.array([11.0, 19.0, 31.0, 38.0, 52.0])

    res = evaluate(y_true, y_pred, task="regression")
    assert "r2" in res
    assert "rmse" in res
    assert "mae" in res
    assert "mape" in res
    assert res["r2"] > 0.95


def test_evaluate_classification():
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.2, 0.3, 0.7, 0.8, 0.9, 0.85])

    res = evaluate(y_true, y_prob, task="classification")
    assert "roc_auc" in res
    assert "pr_auc" in res
    assert "f1" in res
    assert "precision" in res
    assert "recall" in res
    assert "brier" in res
    assert res["roc_auc"] == 1.0


def test_save_metrics(tmp_path):
    metrics = {"r2": 0.92, "rmse": 3200.5}
    out_path = tmp_path / "metrics" / "test_metrics.json"

    save_metrics(metrics, out_path)
    assert out_path.exists()

    with open(out_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["r2"] == 0.92


def test_plots_generation(tmp_path):
    y_true_reg = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
    y_pred_reg = np.array([110.0, 190.0, 310.0, 390.0, 510.0])

    reg_data = {"y_true": y_true_reg, "y_pred": y_pred_reg}

    p_act = plot_predicted_vs_actual(reg_data, out_path=tmp_path / "actual.png")
    assert p_act.exists()

    p_res = plot_residuals(reg_data, out_path=tmp_path / "res.png")
    assert p_res.exists()

    p_err = plot_error_by_segment(
        {"categories": ["A", "B"], "mae": [10.0, 15.0], "rmse": [12.0, 18.0]},
        out_path=tmp_path / "err.png",
    )
    assert p_err.exists()

    p_cov = plot_interval_coverage(
        {"y_true": y_true_reg, "lower": y_pred_reg - 20.0, "upper": y_pred_reg + 20.0},
        out_path=tmp_path / "cov.png",
    )
    assert p_cov.exists()

    y_true_clf = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    y_prob_clf = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.75, 0.8, 0.85, 0.9, 0.95])
    clf_data = {"y_true": y_true_clf, "probs": y_prob_clf}

    p_pr = plot_pr_curve(clf_data, out_path=tmp_path / "pr.png")
    assert p_pr.exists()

    p_roc = plot_roc_curve(clf_data, out_path=tmp_path / "roc.png")
    assert p_roc.exists()

    p_cal = plot_calibration_curve(clf_data, out_path=tmp_path / "cal.png")
    assert p_cal.exists()
