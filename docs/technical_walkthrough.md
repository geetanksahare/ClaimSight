# ClaimSight: Technical Architecture & Actuarial Math Deep Dive

This document details the core mathematical foundations, actuarial modeling choices, and engineering safeguards implemented within ClaimSight.

---

## 1. Conformal Prediction Intervals for Medical Cost Forecasting

### The Problem
Medical cost predictions frequently suffer from heteroscedasticity—errors grow dramatically as predicted expenditure increases. Standard regression models (and even prediction intervals based on assumed Gaussian residuals) severely underestimate uncertainty in high-cost regimes.

### The Mathematical Solution: Split Conformal Prediction
ClaimSight implements **inductive (split) conformal prediction** to guarantee distribution-free, finite-sample marginal coverage:

$$P(Y_{n+1} \in \hat{C}(X_{n+1})) \ge 1 - \alpha$$

Where:
- $\alpha = 0.10$ yields a target **$90\%$ marginal coverage guarantee**.
- Training data is partitioned into:
  - **Train split** ($D_{\text{train}}$): Used to fit the XGBoost / LightGBM regressor $\hat{f}(x)$ on $\log(y+1)$.
  - **Calibration split** ($D_{\text{calib}}$): Held out exclusively to compute non-conformity scores.
  - **Test split** ($D_{\text{test}}$): Evaluated once for final reporting.

### Non-Conformity Score Formulation
Because medical expenditures are strictly positive and right-skewed, the non-conformity score $s_i$ for calibration sample $i$ is calculated in the natural log domain:

$$s_i = |\log(y_i + 1) - \log(\hat{y}_i + 1)|$$

We compute the empirical $(1 - \alpha)(1 + 1/|D_{\text{calib}}|)$ quantile of non-conformity scores:

$$\hat{q} = \text{Quantile}\left(\{s_i\}_{i \in D_{\text{calib}}}, \left\lceil (|D_{\text{calib}}| + 1)(1 - \alpha) \right\rceil / |D_{\text{calib}}|\right)$$

In ClaimSight's calibrated cost pipeline, $\hat{q} = 0.5013$. The resulting prediction interval in dollar terms is:

$$\hat{C}(x) = \left[ \max\left(0, \exp(\hat{z} - \hat{q}) - 1\right), \; \exp(\hat{z} + \hat{q}) - 1 \right]$$

where $\hat{z} = \hat{f}(x)$ is the model's log-space prediction.

**Empirical Verification:** On the held-out test set of 201 samples, ClaimSight achieves **$92.04\%$ empirical coverage**, satisfying the theoretical $90\%$ lower bound.

---

## 2. Zero-Inflated Compound Tweedie Regression

### The Problem: Excess Zeros & Right-Skew in Longitudinal Claims
In prospective member risk modeling, healthcare claims exhibit two properties that break ordinary least squares (OLS) and standard Poisson models:
1. **Zero-Inflation:** A substantial portion of the population incurs exactly $\$0$ in inpatient or emergency claims during any given observation window.
2. **Heavy-Tailed Continuous Expenditure:** Individuals with claims incur continuous, right-skewed positive costs with large variances.

### The Tweedie Compound Poisson-Gamma Distribution
ClaimSight uses a **Tweedie distribution with power parameter $p \in (1, 2)$**, which represents a Poisson sum of Gamma random variables:

$$Y = \sum_{j=1}^{N} X_j, \quad N \sim \text{Poisson}(\lambda), \quad X_j \sim \text{Gamma}(\alpha, \beta)$$

Where:
- $N$ represents the **claim count** (frequency).
- $X_j$ represents the **individual claim size** (severity).
- The variance function satisfies $\text{Var}(Y) = \phi \cdot \mu^p$, where $\mu = E[Y]$.

Using LightGBM's `tweedie` objective with parameter optimization via Optuna ($p \approx 1.5$), ClaimSight models the expected aggregate expenditure directly without requiring two disconnected sub-models.

### Isotonic Probability Calibration
To identify members transitioning into the catastrophic **top decile** (claims $> \$2,790.34$), the classification head outputs probabilities calibrated via **non-parametric isotonic regression**:

$$\min_{m} \sum_{i} (y_i - m(\hat{p}_i))^2 \quad \text{subject to } m(a) \le m(b) \text{ whenever } a \le b$$

This prevents over-confident probabilities, yielding a well-calibrated **Brier score of $0.0934$** and **PR-AUC of $0.1918$** (against a $10.8\%$ base rate).

---

## 3. Explainability Mathematics: TreeSHAP & Counterfactuals

### SHAP (SHapley Additive exPlanations)
To provide local feature attribution that satisfies efficiency, symmetry, dummy, and additivity axioms:

$$\phi_i(x) = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!(|F| - |S| - 1)!}{|F|!} \left[ f(S \cup \{i\}) - f(S) \right]$$

ClaimSight computes exact TreeSHAP values in polynomial time $O(TLD^2)$ using `shap.TreeExplainer`, generating dynamic waterfall plots for every prediction.

### Counterfactual What-If Sandbox
The counterfactual engine computes partial derivatives across modifiable behavioral features (e.g., `smoker`, `bmi`):

$$\Delta_{\text{spend}} = \hat{f}(x_{\text{modified}}) - \hat{f}(x_{\text{baseline}})$$

Coupled with re-computed conformal intervals, this provides actuaries with quantitative bounds on lifestyle intervention value.

---

## 4. Strict Data Leakage Controls

ClaimSight enforces rigorous anti-leakage constraints in `src/claimsight/data/`:
1. **Single-Use Test Splits:** The test set is partitioned upfront before any exploratory data analysis or pipeline execution.
2. **Encapsulated Preprocessing:** All scalers, one-hot encoders, and imputation transformers are fit strictly inside `Pipeline.fit()` on $D_{\text{train}}$. No test or calibration records are ever seen by transformers.
3. **Held-Out Conformal Calibration:** Quantile $\hat{q}$ is calculated on an independent calibration split ($D_{\text{calib}}$) and never on the training or test splits.
4. **Target Cutoff Derivation:** Risk tier thresholds (Low: $\le \$1,020.84$, Moderate: $\le \$1,480.79$, High: $> \$2,350.11$) and high-cost binary thresholds are derived strictly from training set empirical percentiles.

---

## 5. Summary of Tested Metric Benchmarks

```
========================================================================================
Module                     Primary Metric                Secondary Metric       Coverage / Calib
----------------------------------------------------------------------------------------
1. Cost Prediction         R² = 0.9221                   RMSE = $3,360.83       92.04% (q=0.5013)
2. Member Risk             PR-AUC = 0.1918 (1.8x lift)   ROC-AUC = 0.6744       Brier = 0.0934
3. Fraud Anomaly           PR-AUC = 1.0000 (10.4x lift)  ROC-AUC = 1.0000       Threshold = 0.814
========================================================================================
```
