# Changelog

All notable changes to the ClaimSight project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-10-23

### Added
- **Live Streamlit Cloud Deployment:** App deployed to [https://claimsight.streamlit.app/](https://claimsight.streamlit.app/) with automated data bundle fallbacks.
- **Actuarial Modeling Pipelines:**
  - Cost prediction pipeline using log-scale XGBoost/LightGBM with Split Conformal Prediction intervals (90% target coverage, 92.04% test coverage).
  - Member risk stratification using compound Tweedie regression and isotonic calibrated classification (Brier score 0.0934).
  - Provider fraud anomaly triangulation combining supervised classification and unsupervised Isolation Forest behavioral outlier scoring.
- **Fairlearn Demographic Audits:** Automated disparity analysis across protected attributes (`sex`, `age_band`, `region`, `race`) with 20% disparity flags.
- **Explainable AI:** SHAP TreeExplainer waterfall plots and interactive counterfactual what-if simulations.
- **Dual Serving Interfaces:**
  - FastAPI REST service with OpenAPI documentation and Pydantic v2 validation.
  - Streamlit Clinical Observatory with 7 multi-page interactive modules.
- **Documentation & Collateral:** Complete presentation package in `docs/` (`one_pager.md`, `demo_script.md`, `technical_walkthrough.md`, `qna.md`, `cover_note.md`, `launch_checklist.md`).
- **Comprehensive Test Suite:** 71 unit and integration tests with 95% total code coverage and strict Ruff linting.

### Fixed
- Fixed API error mapping from internal 500 to actionable HTTP 503 Service Unavailable when model artifacts are not yet loaded.
- Fixed PyArrow serialization crash in What-If counterfactual analysis by casting mixed categorical mutation values to strings.
- Fixed hardcoded local Windows file paths in model registry pointers (`latest.json`), replacing with portable relative paths.
- Fixed Streamlit `use_container_width=True` deprecation across all pages, migrating to `width="stretch"`.
- Fixed metrics table display in `scripts/update_readme.py` and `6_Model_Performance.py` to extract member Brier calibration scores and base rates.
- Hardened batch scoring against empty CSV uploads and excessive row counts (>50,000 rows).

### Security & Governance
- Embedded strict public/synthetic data regulatory disclaimers prohibiting unmonitored automated underwriting denials.
- Enforced single-use test set splits and encapsulated scikit-learn pipelines to eliminate data leakage.
