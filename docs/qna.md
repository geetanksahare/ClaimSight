# ClaimSight: Technical & Actuarial Interview Q&A Guide

15 tough questions with honest, metric-grounded responses for data science and actuarial technical interviews.

---

### Q1: Why did you use Split Conformal Prediction instead of Quantile Regression or Bootstrapping?
**Answer:** Quantile regression models often suffer from quantile crossing ($\hat{q}_{0.95} < \hat{q}_{0.05}$) and lack finite-sample distribution-free coverage guarantees without large asymptotic sample sizes. Bootstrapping is computationally prohibitive for sub-50ms REST API inference. Split Conformal Prediction provides a mathematically guaranteed coverage property ($P(Y \in \hat{C}(X)) \ge 1 - \alpha$) regardless of data distribution, with $O(1)$ constant-time inference latency. On our test set, ClaimSight achieved **92.04% empirical coverage** against a 90% nominal target ($\hat{q} = 0.5013$).

---

### Q2: Why did you predict on the log scale $\log(y+1)$ rather than raw dollars?
**Answer:** Medical expenditures are heavily right-skewed with extreme positive outliers. Training tree-based algorithms directly on raw dollars makes squared error loss dominated by a tiny fraction of catastrophic claims, leading to poor calibration and high variance across typical claims. Fitting on the log scale stabilizes variance, transforms multiplicative effects into additive ones, and produces tighter, symmetric non-conformity distributions.

---

### Q3: On the Member Risk module, your test $R^2$ is only 0.0409. Isn't that very low?
**Answer:** Yes, and this is standard for longitudinal healthcare claims prediction. Individual future healthcare utilization is inherently noisy—a healthy individual may suffer an unexpected trauma or sudden acute illness. Attempting to predict the exact dollar amount 12 months out on a zero-inflated synthetic Medicare population is statistically ill-posed.  
Instead of relying on $R^2$, we focus on **actuarial risk tiering** and **calibrated top-decile classification**. Our calibrated classifier achieves a **PR-AUC of 0.1918**—an **80% relative lift over the 10.77% base rate**—with a well-calibrated **Brier score of 0.0934**. For clinical care management, identifying members at elevated risk of catastrophic spend is dramatically more actionable than predicting exact dollars.

---

### Q4: In your Fraud Detection module, your test PR-AUC and ROC-AUC are 1.0000. Is there data leakage?
**Answer:** We conducted an exhaustive data leakage audit. All preprocessing scalers and feature transformers are fit strictly inside `scikit-learn` Pipelines on training folds. The 1.0000 PR-AUC reflects the strong synthetic signal separation in the CMS SynPUF provider data—providers with simulated fraudulent behavior exhibit extreme, distinct anomalies in claim volume and inpatient ratios. In a true production setting with noisy, adversarial fraud labels, PR-AUC would be lower. To counteract synthetic label reliance, ClaimSight couples the supervised classifier with an **unsupervised Isolation Forest**, scoring behavioral anomaly distance independently of ground-truth labels.

---

### Q5: How do you prevent data leakage during conformal calibration?
**Answer:** We strictly split the dataset into three non-overlapping subsets: **Train (60%)**, **Calibration (20%)**, and **Test (20%)**. The base model and transformers are trained exclusively on the Train split. The conformal non-conformity quantile $\hat{q}$ is computed exclusively on the Calibration split. The Test split is evaluated only once at the very end to verify empirical coverage and is never seen during training or tuning.

---

### Q6: Why did you choose a Tweedie distribution for member risk modeling?
**Answer:** Healthcare claims contain a structural point mass at zero (members with no claims) coupled with continuous, right-skewed positive expenditure for claimants. This matches a compound Poisson-Gamma process. The Tweedie distribution parameterized with power $p \in (1, 2)$ naturally accommodates this dual zero-inflation and positive skew in a unified objective without requiring a disconnected two-stage hurdle model.

---

### Q7: How does your system handle algorithmic bias and demographic disparity?
**Answer:** We integrate **Fairlearn** into automated reporting. We evaluate disparity across sex, age bands, region, and race. For example, in our cost model, the MAE gap between male and female cohorts is **$285.38 (18.8% relative gap)**, which passes our 20% parity threshold. However, age cohorts show a **$1,371.32 (90.5%)** gap because older cohorts experience biologically higher medical variance. Rather than concealing this, ClaimSight surfaces transparent disparity flags in both the API and the Streamlit Observatory.

---

### Q8: What happens if an API request contains input features outside the training distribution?
**Answer:** The inference engine checks numerical features against the min/max bounding box from the training dataset. If a feature (e.g., `age=105` or `bmi=70`) exceeds training boundaries, the API flags `in_training_range: false` and emits explicit actionable warnings in the response JSON, alerting underwriters to potential out-of-distribution extrapolation risk.

---

### Q9: Can an insurance company use this system to automatically deny claims or cancel policies?
**Answer:** Absolutely not. We include explicit disclaimers in the UI, API responses, and documentation. ClaimSight is an actuarial intelligence and decision-support tool. Using automated predictive models for adverse underwriting actions or claims denial violates ACA Section 1557 regulations and actuarial standards of practice. It is built to assist clinical case managers and fraud audit prioritization.

---

### Q10: How do you achieve sub-50ms latency in your FastAPI service?
**Answer:** All model artifacts (LightGBM/XGBoost trees, preprocessors, conformal non-conformity quantiles) are pre-loaded into memory during FastAPI startup via a lifespan handler. Predictions use vector-optimized C++ backends. SHAP tree explainers are loaded lazily or on-demand to keep primary scoring endpoints lightning fast.

---

### Q11: How did you validate that your Streamlit app doesn't crash on edge cases?
**Answer:** We wrote comprehensive test suites with `pytest` (71 passing tests, 95% total code coverage). Specifically:
1. `tests/app/test_batch_validation.py` tests empty CSVs, malformed headers, and files exceeding 50,000 rows.
2. `tests/inference/test_cost_inference.py` tests boundary conditions, missing attributes, and zero-variance scenarios.
3. PyArrow type serialization was hardened to prevent mixed-type counterfactual dataframe crashes.

---

### Q12: Why did you build an unsupervised Isolation Forest alongside the supervised fraud classifier?
**Answer:** Real-world insurance fraud labels are notoriously incomplete—investigators only catch a fraction of fraudulent schemes, creating strong selection bias in labeled data. By running an unsupervised Isolation Forest alongside the supervised model, we detect novel, previously unseen billing pattern anomalies that a supervised classifier trained on historic fraud labels would overlook.

---

### Q13: What is the significance of the Brier score in the Member Risk module?
**Answer:** ROC-AUC measures ranking discrimination, but health plans allocating clinical budgets need well-calibrated probabilities (e.g., a 20% predicted probability should correspond to 2 out of 10 members actually incurring catastrophic spend). Our model achieves a **Brier score of 0.0934**, validated via isotonic regression, proving that the output probabilities reflect true frequencies.

---

### Q14: How does the What-If Counterfactual feature work under the hood?
**Answer:** Given a baseline member profile and a set of proposed behavioral modifications (e.g., `smoker: yes -> no`), the engine creates a mutated feature dictionary, applies the identical trained feature transformation pipeline, runs inference, and computes both point cost delta and new conformal bounds. Categorical changes are string-casted to guarantee safe Arrow serialization in Streamlit.

---

### Q15: How would you deploy ClaimSight in a production hospital or insurer environment?
**Answer:** ClaimSight is packaged with a multi-stage `Dockerfile` and `docker-compose.yml`. In enterprise settings:
1. The FastAPI microservice runs inside Kubernetes behind an API Gateway with JWT authentication and Prometheus monitoring.
2. Training pipelines execute via scheduled Airflow or Kubeflow DAGs with MLflow experiment tracking.
3. Streamlit runs behind an enterprise SSO (Okta/Azure AD) for clinical care coordinators and SIU investigators.
