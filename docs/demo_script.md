# ClaimSight: 3-Minute Live Technical Demo Script

**Audience:** Hiring managers, Chief Actuaries, VP of Data Science, or Lead ML Engineers.  
**Platform:** [claimsight.streamlit.app](https://claimsight.streamlit.app/) (or local `streamlit run src/claimsight/app/Home.py`).  
**Time Limit:** Exactly 3 minutes (180 seconds).

---

### Step 1: 0:00 - 0:30 — Overview & Architecture (Home Page)
1. **Action:** Open `Home.py` in browser.
2. **Talking Points:**
   > "Most healthcare cost models suffer from two critical flaws: they offer bare point predictions with zero uncertainty quantification, and they treat pricing, member risk, and fraud as disconnected silos.  
   > ClaimSight brings actuarial rigor, split-conformal inference, and explainable AI together under an enterprise-ready FastAPI and Streamlit interface. On this landing page, we see our three active models loaded into the unified registry, complete with health checks and data lineage."

---

### Step 2: 0:30 - 1:15 — Conformal Pricing & What-If Simulation (`1_Cost_Estimator.py`)
1. **Action:** Click **Cost Estimator** in the sidebar navigation.
2. **Inputs:**
   - **Age:** 45
   - **Sex:** Male
   - **BMI:** 32.0 (Obese range)
   - **Children:** 1
   - **Smoker:** Yes
   - **Region:** Northwest
3. **Observation:**
   - The system displays the predicted annual charge (~$36,000–$40,000) alongside a **90% Conformal Prediction Interval** ($[y_{\text{lower}}, y_{\text{upper}}]$).
   - Show the **SHAP Waterfall Plot** demonstrating that `smoker=yes` is the primary positive cost driver.
4. **Interactive Action (What-If):**
   - In the "What-If Counterfactual Sandbox", switch `smoker` from **yes** to **no**.
   - Show the immediate delta: annual expected charges drop by over ~$25,000.
5. **Talking Points:**
   > "Notice we don't just give an underwriter an ungrounded point estimate. We provide a mathematically proven split-conformal interval with 92% empirical test coverage. If this individual stops smoking, the counterfactual engine recalculates the new distribution in real-time."

---

### Step 3: 1:15 - 1:50 — Longitudinal Member Risk Tiering (`3_Member_Risk.py`)
1. **Action:** Click **Member Risk** in the sidebar.
2. **Inputs:**
   - Choose or enter a prospective beneficiary: Age 68, Female, Chronic Flags: Diabetes + Heart Failure, Prior Inpatient Spend: $2,500.
3. **Observation:**
   - Display the expected prospective expenditure modeled via **Compound Tweedie Regression** (handling zero-inflation).
   - Display the **Calibrated High-Risk Probability** (Brier score $= 0.0934$) and assigned Risk Tier: **High Risk (Top 10%)**.
4. **Talking Points:**
   > "Healthcare claims follow a zero-inflated, compound Poisson-Gamma distribution. Traditional linear models break here. We use a Tweedie regressor paired with isotonic probability calibration to identify members at risk of catastrophic hospitalization before costs occur."

---

### Step 4: 1:50 - 2:30 — Provider Anomaly & Fraud Triangulation (`4_Anomaly_Review.py`)
1. **Action:** Click **Anomaly Review** in the sidebar.
2. **Observation:**
   - The provider leaderboard shows providers ranked by **Composite Fraud Priority Score**.
   - Select the top anomalous provider (`PRV55462` or top row) to open the deep-dive radar and clinical profile.
3. **Talking Points:**
   > "Retrospective fraud audits fail because labeled fraud data is heavily biased. We triangulate an Optuna-tuned classifier with an unsupervised Isolation Forest. The result is a behavioral composite score highlighting extreme inpatient ratios, high claims velocity, and anomalous physician credential concentrations for SIU investigation."

---

### Step 5: 2:30 - 3:00 — Actuarial Governance & Fairness (`6_Model_Performance.py`)
1. **Action:** Click **Model Performance** in the sidebar.
2. **Observation:**
   - Review live ROC and PR curves, Calibration curves, and the **Fairlearn Demographic Disparity Audit** table.
3. **Closing Talking Points:**
   > "Everything in ClaimSight is audited for demographic fairness. If error rates across age cohorts or regions exceed our 20% relative disparity threshold, the system flags it automatically. The platform is fully tested with 95% test coverage, containerized with Docker, and backed by a FastAPI microservice."

---

### Fallback Contingency Plan
- **If the cloud app takes time to wake from idle:** Show local Streamlit app at `http://localhost:8501` or switch to the Swagger REST docs at `http://localhost:8000/docs` to demonstrate live cURL API requests.
- **If asked about real clinical data:** Emphasize that the architecture is production-ready and built on CMS synthetic Medicare longitudinal files (SynPUF) to comply with HIPAA and privacy regulations.
