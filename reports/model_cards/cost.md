# Model Card: Cost Intelligence Model

Built on public/synthetic data. Not for real pricing, underwriting or claims decisions.

## 1. Overview
- **Model Identifier:** `cost`
- **Version:** `20261021_152517`
- **Model Type:** `XGBRegressor`
- **Date Generated:** `2026-10-21 15:25 UTC`
- **Git Commit:** `81a39def9788dd1d3a2ed2b6daecc5cf9fdfbb5a`
- **Scikit-Learn Version:** `1.7.2`

## 2. Intended Use & Scope
- **Intended Use:** Benchmark predictive modeling, demographic disparity auditing, and explainability exploration on public synthetic healthcare insurance claims data.
- **Out-of-Scope Use:** Not for clinical diagnosis, commercial insurance underwriting, individual pricing, or automated claims denial.

## 3. Training Data
- **Dataset Source:** Public Kaggle / CMS Public Use File
- **Row Count:** 935 training samples
- **Licence:** Public Open Access / Synthetic Public Use File
- **Data Statement:** Built on public/synthetic data. Not for real pricing, underwriting or claims decisions.

## 4. Features & Split Strategy
- **Feature Set:** age, bmi, children, sex, smoker, region, age_band, bmi_category, is_obese, smoker_flag, sex_flag, has_children, smoker_x_obese, smoker_x_age, bmi_x_smoker, family_size_bucket
- **Split Strategy:** Stratified on smoker, 70/15/15 (train/calib/test)
- **Global Random Seed:** 42

## 5. Quantitative Evaluation
All metrics read directly from code evaluation reports (`reports/metrics/cost.json`).

| Metric | Score |
|---|---|


| rmse | 3360.8325 |



| mae | 1514.3828 |



| r2 | 0.9221 |



| median_ae | 663.958 |



| mape | 16.1526 |



| n_samples | 201 |





| conformal_q | 0.5013 |



| conformal_coverage | 0.9204 |



| target_coverage | 0.9 |



| best_model | xgboost |




### Conformal Prediction Intervals
- **Log Nonconformity Quantile (q):** 0.501251864349415
- **Empirical Coverage on Test Set:** 92.0% (target: 90%)


## 6. Fairness & Subgroup Parity
- **Audit Tool:** Fairlearn `MetricFrame`
- **Summary:**

- `sex`: Disparity gap = 285.3843 (Relative gap = 18.8%, Status: Passed)

- `region`: Disparity gap = 389.2052 (Relative gap = 25.7%, Status: FLAGGED)

- `age_band`: Disparity gap = 1371.3164 (Relative gap = 90.5%, Status: FLAGGED)


## 7. Known Limitations & Ethical Considerations
The cost dataset is a small US sample with four regions; SynPUF is synthetic Medicare data (older population, altered correlations); fraud labels are provider-level with about 9 percent positives; none of these datasets represent any specific country's market or any real insurer; associations are not causal.

## 8. Reproducibility
To reproduce this model training run from scratch:
```bash
make train-cost
```