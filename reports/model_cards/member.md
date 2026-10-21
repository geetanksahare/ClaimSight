# Model Card: Member Intelligence Model

Built on public/synthetic data. Not for real pricing, underwriting or claims decisions.

## 1. Overview
- **Model Identifier:** `member`
- **Version:** `20261021_153453`
- **Model Type:** `dict`
- **Date Generated:** `2026-10-21 15:34 UTC`
- **Git Commit:** `4bd7a7cadf420803c12dd2cdfb251e3ad92c1e5c`
- **Scikit-Learn Version:** `1.7.2`

## 2. Intended Use & Scope
- **Intended Use:** Benchmark predictive modeling, demographic disparity auditing, and explainability exploration on public synthetic healthcare insurance claims data.
- **Out-of-Scope Use:** Not for clinical diagnosis, commercial insurance underwriting, individual pricing, or automated claims denial.

## 3. Training Data
- **Dataset Source:** Public Kaggle / CMS Public Use File
- **Row Count:** 3148 training samples
- **Licence:** Public Open Access / Synthetic Public Use File
- **Data Statement:** Built on public/synthetic data. Not for real pricing, underwriting or claims decisions.

## 4. Features & Split Strategy
- **Feature Set:** age_at_year_end, sex, race, esrd, SP_ALZHDMTA, SP_CHF, SP_CHRNKIDN, SP_CNCR, SP_COPD, SP_DEPRESSN, SP_DIABETES, SP_ISCHMCHT, SP_OSTEOPRS, SP_RA_OA, SP_STRKETIA, chronic_count, prior_cost_ip, prior_cost_op, prior_cost_car, prior_total_cost, prior_total_claims, cost_per_claim, cost_change, has_history, ip_days, ip_claims, op_claims, carrier_claims
- **Split Strategy:** GroupShuffleSplit (80/20) on member_id
- **Global Random Seed:** 42

## 5. Quantitative Evaluation
All metrics read directly from code evaluation reports (`reports/metrics/member.json`).

| Metric | Score |
|---|---|


| regression | {'rmse': 2502.8334, 'mae': 1171.4039, 'r2': 0.0409, 'median_ae': 675.8351, 'mape': 5422.8651, 'n_samples': 780} |



| classification | {'roc_auc': 0.6744, 'pr_auc': 0.1918, 'f1': 0.023, 'precision': 0.3333, 'recall': 0.0119, 'brier': 0.0934, 'threshold_used': 0.5, 'threshold_note': 'custom', 'base_rate': 0.1077, 'precision_at_top_k': 0.2179, 'recall_at_top_k': 0.2024, 'confusion_matrix': [[694, 2], [83, 1]], 'n_samples': 780} |



| high_cost_threshold | 2790.34 |



| tier_cutoffs | [1020.84, 1480.79, 2350.11] |



| tier_quantiles | [0.5, 0.8, 0.95] |



| n_train | 3148 |



| n_test | 780 |



| test_r2 | 0.0409 |



| test_rmse | 2502.8334 |



| pr_auc | 0.1918 |



| roc_auc | 0.6744 |



| f1 | 0.023 |





## 6. Fairness & Subgroup Parity
- **Audit Tool:** Fairlearn `MetricFrame`
- **Summary:**

- `sex`: Disparity gap = 0.0081 (Relative gap = 210.6%, Status: FLAGGED)

- `race`: Disparity gap = 0.0047 (Relative gap = 122.2%, Status: FLAGGED)

- `age_band`: Disparity gap = 0.0067 (Relative gap = 174.2%, Status: FLAGGED)


## 7. Known Limitations & Ethical Considerations
The cost dataset is a small US sample with four regions; SynPUF is synthetic Medicare data (older population, altered correlations); fraud labels are provider-level with about 9 percent positives; none of these datasets represent any specific country's market or any real insurer; associations are not causal.

## 8. Reproducibility
To reproduce this model training run from scratch:
```bash
make train-member
```