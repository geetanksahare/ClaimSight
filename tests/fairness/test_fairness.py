import numpy as np
import pandas as pd

from claimsight.evaluation.fairness_audit import fairness_audit


def test_fairness_metric_frame_gaps():
    # 70 rows: 35 male, 35 female (>= 30 per group to pass low_n filter)
    y_true = np.array([100.0] * 35 + [200.0] * 35)
    # Predicted with deliberate group disparity:
    # Male: pred = 110 (MAE = 10)
    # Female: pred = 250 (MAE = 50)
    y_pred = np.array([110.0] * 35 + [250.0] * 35)

    sensitive = pd.DataFrame({"sex": ["male"] * 35 + ["female"] * 35})

    audit = fairness_audit(
        y_true=y_true,
        y_pred=y_pred,
        sensitive=sensitive,
        task="regression",
        relative_gap_threshold=0.20,
        module="test_mod",
    )

    sex_audit = audit["columns"]["sex"]
    # Overall MAE is (20*10 + 20*50) / 40 = 30.0
    assert sex_audit["overall_metric"] == 30.0

    # Max MAE (50) - Min MAE (10) = 40.0
    assert sex_audit["max_min_gap"] == 40.0
    # Relative gap is 40.0 / 30.0 = 1.3333 (> 0.20 threshold -> flagged)
    assert sex_audit["disparity_flag"] is True


def test_fairness_low_n_handling():
    # Group with fewer than 30 rows should be marked low_n and excluded from disparity flag
    y_true = np.array([100.0] * 40 + [200.0] * 10)
    y_pred = np.array([100.0] * 40 + [500.0] * 10)  # group B has huge error but N=10 (<30)

    sensitive = pd.DataFrame({"group": ["A"] * 40 + ["B"] * 10})

    audit = fairness_audit(
        y_true=y_true,
        y_pred=y_pred,
        sensitive=sensitive,
        task="regression",
        relative_gap_threshold=0.20,
        module="test_low_n",
    )

    group_data = audit["columns"]["group"]
    assert group_data["groups"]["B"]["low_n"] is True
    assert group_data["groups"]["A"]["low_n"] is False
    # Only 1 valid group with count >= 30, so disparity gap cannot be calculated between groups
    assert group_data["disparity_flag"] is False
