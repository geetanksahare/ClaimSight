import json
from pathlib import Path


def generate_metrics_markdown() -> str:
    metrics_dir = Path("reports/metrics")

    # 1. Cost Metrics
    cost_file = metrics_dir / "cost.json"
    cost_data = json.loads(cost_file.read_text(encoding="utf-8")) if cost_file.exists() else {}

    # 2. Member Metrics
    member_file = metrics_dir / "member.json"
    member_data = (
        json.loads(member_file.read_text(encoding="utf-8")) if member_file.exists() else {}
    )

    # 3. Fraud Metrics
    fraud_file = metrics_dir / "fraud.json"
    fraud_data = json.loads(fraud_file.read_text(encoding="utf-8")) if fraud_file.exists() else {}

    # 4. Fairness Metrics
    f_cost_file = metrics_dir / "fairness_cost.json"
    f_cost = json.loads(f_cost_file.read_text(encoding="utf-8")) if f_cost_file.exists() else {}

    f_member_file = metrics_dir / "fairness_member.json"
    f_member = (
        json.loads(f_member_file.read_text(encoding="utf-8")) if f_member_file.exists() else {}
    )

    f_fraud_file = metrics_dir / "fairness_fraud.json"
    f_fraud = json.loads(f_fraud_file.read_text(encoding="utf-8")) if f_fraud_file.exists() else {}

    md = []
    md.append("### Quantitative Model Performance Leaderboard\n")
    md.append(
        "| Module | Task Type | Champion Model | Primary Test Metric | Secondary Metric | Conformal / Calibration |"
    )
    md.append("|---|---|---|---|---|---|")

    # Cost Row
    c_r2 = f"{cost_data.get('r2', 0.0):.4f}" if cost_data else "N/A"
    c_rmse = f"${cost_data.get('rmse', 0.0):,.2f}" if cost_data else "N/A"
    c_cov = (
        f"{cost_data.get('conformal_coverage', 0.0) * 100:.1f}% coverage (q={cost_data.get('conformal_q', 0.0):.3f})"
        if cost_data
        else "N/A"
    )
    c_best = cost_data.get("best_model", "xgboost")
    md.append(
        f"| **1. Cost Prediction** | Regression (Log-scale) | `{c_best}` | R² = {c_r2} | RMSE = {c_rmse} | {c_cov} |"
    )

    # Member Row
    m_pr = (
        f"{member_data.get('pr_auc', 0.0):.4f} (Base: {member_data.get('base_rate', 0.1) * 100:.1f}%)"
        if member_data
        else "N/A"
    )
    m_roc = f"ROC-AUC = {member_data.get('roc_auc', 0.0):.4f}" if member_data else "N/A"
    m_brier = f"Brier = {member_data.get('brier', 0.0):.4f}" if member_data else "N/A"
    m_best = member_data.get("best_model", "lightgbm")
    md.append(
        f"| **2. Member Risk** | Compound Tweedie + Isotonic | `{m_best}` | PR-AUC = {m_pr} | {m_roc} | {m_brier} |"
    )

    # Fraud Row
    f_pr = (
        f"{fraud_data.get('pr_auc', 0.0):.4f} ({fraud_data.get('pr_auc_vs_base_rate_ratio', 1.0):.1f}x lift)"
        if fraud_data
        else "N/A"
    )
    f_roc = f"ROC-AUC = {fraud_data.get('roc_auc', 0.0):.4f}" if fraud_data else "N/A"
    f_thresh = (
        f"Threshold = {fraud_data.get('threshold_used', 0.5):.3f} (F1 = {fraud_data.get('f1', 0.0):.3f})"
        if fraud_data
        else "N/A"
    )
    f_best = fraud_data.get("best_model", "logistic_regression")
    md.append(
        f"| **3. Fraud Detection** | Supervised + Isolation Forest | `{f_best}` | PR-AUC = {f_pr} | {f_roc} | {f_thresh} |"
    )

    md.append("\n### Demographic Disparity & Fairness Audit Summary\n")
    md.append(
        "| Module | Sensitive Feature | Metric Evaluated | Max-Min Gap | Relative Gap | Audit Status |"
    )
    md.append("|---|---|---|---|---|---|")

    # Fairness Cost Rows
    if f_cost and "columns" in f_cost:
        for feat, audit in f_cost["columns"].items():
            status = (
                "⚠️ Disparity Flag (>20%)" if audit.get("disparity_flag") else "✅ Passed (≤20%)"
            )
            rel = f"{audit.get('relative_gap', 0.0) * 100:.1f}%"
            gap = f"${audit.get('max_min_gap', 0.0):,.2f}"
            md.append(f"| Cost Estimation | `{feat}` | MAE | {gap} | {rel} | {status} |")

    # Fairness Member Rows
    if f_member and "columns" in f_member:
        for feat, audit in f_member["columns"].items():
            status = (
                "⚠️ Disparity Flag (>20%)" if audit.get("disparity_flag") else "✅ Passed (≤20%)"
            )
            rel = f"{audit.get('relative_gap', 0.0) * 100:.1f}%"
            gap = f"{audit.get('max_min_gap', 0.0):.3f}"
            md.append(f"| Member Risk | `{feat}` | Selection Rate | {gap} | {rel} | {status} |")

    # Fairness Fraud Rows
    if f_fraud and "columns" in f_fraud:
        for feat, audit in f_fraud["columns"].items():
            status = (
                "⚠️ Disparity Flag (>20%)" if audit.get("disparity_flag") else "✅ Passed (≤20%)"
            )
            rel = f"{audit.get('relative_gap', 0.0) * 100:.1f}%"
            gap = f"{audit.get('max_min_gap', 0.0):.3f}"
            md.append(f"| Fraud Anomaly | `{feat}` | Selection Rate | {gap} | {rel} | {status} |")

    return "\n".join(md)


def update_readme():
    readme_path = Path("README.md")
    if not readme_path.exists():
        print("README.md does not exist yet.")
        return

    content = readme_path.read_text(encoding="utf-8")
    metrics_md = generate_metrics_markdown()

    start_tag = "<!-- AUTOGENERATED_METRICS_START -->"
    end_tag = "<!-- AUTOGENERATED_METRICS_END -->"

    if start_tag in content and end_tag in content:
        prefix = content.split(start_tag)[0]
        suffix = content.split(end_tag)[1]
        new_content = f"{prefix}{start_tag}\n\n{metrics_md}\n\n{end_tag}{suffix}"
        readme_path.write_text(new_content, encoding="utf-8")
        print("Successfully updated README.md with live metrics from reports/metrics/*.json.")
    else:
        print("Placeholder tags not found in README.md.")


if __name__ == "__main__":
    update_readme()
