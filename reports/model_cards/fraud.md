# Model Card: Fraud Intelligence Model

Built on public/synthetic data. Not for real pricing, underwriting or claims decisions.

## 1. Overview
- **Model Identifier:** `fraud`
- **Version:** `20261021_154402`
- **Model Type:** `dict`
- **Date Generated:** `2026-10-21 15:44 UTC`
- **Git Commit:** `482b8a5de7c79bb1cc13af9878295e2bf41116d6`
- **Scikit-Learn Version:** `1.7.2`

## 2. Intended Use & Scope
- **Intended Use:** Benchmark predictive modeling, demographic disparity auditing, and explainability exploration on public synthetic healthcare insurance claims data.
- **Out-of-Scope Use:** Not for clinical diagnosis, commercial insurance underwriting, individual pricing, or automated claims denial.

## 3. Training Data
- **Dataset Source:** Public Kaggle / CMS Public Use File
- **Row Count:** 960 training samples
- **Licence:** Public Open Access / Synthetic Public Use File
- **Data Statement:** Built on public/synthetic data. Not for real pricing, underwriting or claims decisions.

## 4. Features & Split Strategy
- **Feature Set:** n_claims, n_inpatient, n_outpatient, inpatient_share, n_unique_beneficiaries, claims_per_beneficiary, reimb_sum, reimb_mean, reimb_median, reimb_max, reimb_std, deductible_mean, n_unique_attending, n_unique_operating, attending_missing_share, operating_missing_share, top_attending_claim_share, mean_los, mean_claim_duration, mean_n_diagnosis_codes, mean_n_procedure_codes, n_unique_diag_codes, bene_mean_age, bene_chronic_mean, bene_renal_share, bene_mean_partA_months, share_claims_same_day_multiple
- **Split Strategy:** StratifiedKFold (80/20) on is_fraud
- **Global Random Seed:** 42

## 5. Quantitative Evaluation
All metrics read directly from code evaluation reports (`reports/metrics/fraud.json`).

| Metric | Score |
|---|---|


| roc_auc | 1.0 |



| pr_auc | 1.0 |



| f1 | 1.0 |



| precision | 1.0 |



| recall | 1.0 |



| brier | 0.0 |



| threshold_used | 0.814 |



| threshold_note | custom |



| base_rate | 0.0958 |



| precision_at_top_k | 0.9583 |



| recall_at_top_k | 1.0 |





| n_samples | 240 |



| positive_base_rate | 0.0958 |



| pr_auc_vs_base_rate_ratio | 10.43 |



| best_model | logistic_regression |





## 6. Fairness & Subgroup Parity
- **Audit Tool:** Fairlearn `MetricFrame`
- **Summary:**

- `inpatient_share_tercile`: Disparity gap = 0.0758 (Relative gap = 79.1%, Status: FLAGGED)

- `provider_size_tercile`: Disparity gap = 0.2875 (Relative gap = 300.0%, Status: FLAGGED)


## 7. Known Limitations & Ethical Considerations
The cost dataset is a small US sample with four regions; SynPUF is synthetic Medicare data (older population, altered correlations); fraud labels are provider-level with about 9 percent positives; none of these datasets represent any specific country's market or any real insurer; associations are not causal.

## 8. Reproducibility
To reproduce this model training run from scratch:
```bash
make train-fraud
```