import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from claimsight.api.schemas import CostInput, FlagRequest, MemberInput, WhatIfRequest
from claimsight.api.security import verify_api_key
from claimsight.config import get_settings
from claimsight.explain.explain_prediction import explain_prediction
from claimsight.inference.flag_anomalies import flag_anomalies
from claimsight.inference.predict_cost import predict_cost
from claimsight.inference.score_member_risk import score_member_risk
from claimsight.inference.what_if import what_if
from claimsight.logging_config import get_logger
from claimsight.models.registry import load_model

logger = get_logger(__name__)

loaded_models_registry: dict[str, dict[str, Any]] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event to preload production models into memory upon API startup."""
    logger.info("Initializing ClaimSight REST API and warming model caches...")
    for module_name in ["cost", "member", "fraud"]:
        try:
            _, meta = load_model(module_name)
            loaded_models_registry[module_name] = meta
            logger.info(
                "Successfully loaded %s model (version: %s)", module_name, meta.get("version")
            )
        except Exception as e:
            logger.warning("Could not preload %s model at startup: %s", module_name, e)
    yield
    logger.info("Shutting down ClaimSight REST API...")


app = FastAPI(
    title="ClaimSight REST API",
    description="Actuarial, member risk and claims fraud anomaly intelligence service.",
    version="1.0.0",
    lifespan=lifespan,
)

cfg = get_settings()

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=cfg.api.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_id_and_logging_middleware(request: Request, call_next):
    """Assign or preserve unique X-Request-ID header on all incoming requests."""
    req_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    response: Response = await call_next(request)
    response.headers["X-Request-ID"] = req_id
    return response


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Return structured JSON error for Pydantic schema validation failures."""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "validation_error",
            "message": "The request body failed schema validation.",
            "details": exc.errors(),
        },
    )


@app.get("/health", tags=["System"])
async def health_check():
    """Health check endpoint reporting server status and loaded models."""
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "models_loaded": list(loaded_models_registry.keys()),
        "disclaimer": "Built on public/synthetic data. Not for real pricing, underwriting or claims decisions.",
    }


@app.get("/v1/models", tags=["System"], dependencies=[Depends(verify_api_key)])
async def get_models():
    """Return metadata and registration details for all active models."""
    models_meta = {}
    for module in ["cost", "member", "fraud"]:
        try:
            _, meta = load_model(module)
            models_meta[module] = meta
        except Exception as e:
            models_meta[module] = {"status": "unavailable", "error": str(e)}
    return {"models": models_meta}


def _handle_api_error(module: str, exc: Exception) -> None:
    """Map missing model errors to HTTP 503 Service Unavailable, other exceptions to HTTP 500."""
    err_msg = str(exc)
    if isinstance(exc, FileNotFoundError) or "Run 'claimsight train-" in err_msg:
        logger.warning("Model service unavailable for %s: %s", module, err_msg)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model service unavailable: {err_msg}",
        )
    logger.error("%s endpoint error: %s", module, err_msg)
    raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=err_msg)


@app.post("/v1/cost/predict", tags=["Cost Prediction"], dependencies=[Depends(verify_api_key)])
async def predict_individual_cost(payload: CostInput):
    """Predict annual healthcare charges with 90% split conformal prediction interval."""
    try:
        result = predict_cost(payload.model_dump())
        return result
    except Exception as e:
        _handle_api_error("cost", e)


@app.post("/v1/cost/explain", tags=["Cost Prediction"], dependencies=[Depends(verify_api_key)])
async def explain_cost(payload: CostInput, top_k: int = 5):
    """Provide local SHAP feature attributions and natural language explanation for cost prediction."""
    try:
        df_row = pd.DataFrame([payload.model_dump()])
        result = explain_prediction("cost", df_row, top_k=top_k)
        return result
    except Exception as e:
        _handle_api_error("cost", e)


@app.post("/v1/cost/what-if", tags=["Cost Prediction"], dependencies=[Depends(verify_api_key)])
async def what_if_analysis(payload: WhatIfRequest):
    """Calculate counterfactual prediction delta for modified feature inputs."""
    try:
        base_dict = payload.base.model_dump()
        result = what_if(base_dict, payload.changes)
        return result
    except Exception as e:
        _handle_api_error("cost", e)


@app.post("/v1/members/score", tags=["Member Risk"], dependencies=[Depends(verify_api_key)])
async def score_single_member(payload: MemberInput):
    """Score individual member prospective 12-month healthcare cost and high-cost risk tier."""
    try:
        df_row = pd.DataFrame([payload.model_dump()])
        scored_df = score_member_risk(df_row)
        return scored_df.iloc[0].to_dict()
    except Exception as e:
        _handle_api_error("member", e)


@app.post("/v1/members/score-batch", tags=["Member Risk"], dependencies=[Depends(verify_api_key)])
async def score_batch_members(payload: list[MemberInput]):
    """Score batch of members up to configured limit with Tweedie regression and isotonic calibrated classification."""
    max_rows = cfg.api.max_batch_rows
    if len(payload) > max_rows:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE
            if hasattr(status, "HTTP_413_CONTENT_TOO_LARGE")
            else 413,
            detail=f"Batch size of {len(payload)} exceeds maximum permitted size of {max_rows} rows.",
        )
    try:
        df = pd.DataFrame([m.model_dump() for m in payload])
        scored_df = score_member_risk(df)
        return scored_df.to_dict(orient="records")
    except Exception as e:
        _handle_api_error("member", e)


@app.post(
    "/v1/providers/flag", tags=["Fraud & Anomaly Review"], dependencies=[Depends(verify_api_key)]
)
async def flag_provider_anomalies(payload: FlagRequest):
    """Flag healthcare providers for audit review blending supervised fraud risk and unsupervised anomaly scores."""
    try:
        df = pd.DataFrame([p.model_dump() for p in payload.providers])
        ranked_df = flag_anomalies(df, top_n=payload.top_n)
        return ranked_df.to_dict(orient="records")
    except Exception as e:
        _handle_api_error("fraud", e)
