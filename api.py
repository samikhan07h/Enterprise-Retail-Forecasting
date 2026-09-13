"""
Retail Forecasting API
-----------------------
Serves multi-step-ahead forecasts from whichever model won the most
recent benchmark run (see pipeline.py -> evaluate_models()).

Design note: this endpoint does NOT accept live sales data. The training
series used in the last dashboard run is already baked into
models/forecast_artifact.pkl, and forecasts simply continue from where
that series ends. Re-run the pipeline from the dashboard to refresh it.
"""

import os
import logging
from typing import List

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from pipeline import forecast_from_artifact

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("retail-forecasting-api")

ARTIFACT_PATH = os.environ.get("ARTIFACT_PATH", "models/forecast_artifact.pkl")

app = FastAPI(
    title="Retail Forecasting API",
    description="Serves forecasts from the current best-performing model.",
    version="2.0.0",
)

# Tighten allow_origins to your actual frontend domain(s) in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_artifact = None


def get_artifact():
    """Lazily load and cache the artifact so it's only read from disk once."""
    global _artifact
    if _artifact is None:
        if not os.path.exists(ARTIFACT_PATH):
            raise HTTPException(
                status_code=503,
                detail=(
                    f"No trained forecast artifact found at '{ARTIFACT_PATH}'. "
                    "Run the pipeline from the dashboard first."
                ),
            )
        _artifact = joblib.load(ARTIFACT_PATH)
    return _artifact


class ForecastRequest(BaseModel):
    horizon: int = Field(7, ge=1, le=90, description="Number of future periods to forecast.")


class ForecastResponse(BaseModel):
    model: str
    predictions: List[float]


@app.get("/")
def root():
    return {"message": "Retail Forecasting API is running", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok", "artifact_available": os.path.exists(ARTIFACT_PATH)}


@app.post("/predict", response_model=ForecastResponse)
def predict(request: ForecastRequest):
    artifact = get_artifact()
    try:
        preds = forecast_from_artifact(artifact, request.horizon)
        return ForecastResponse(
            model=artifact["best_model_name"],
            predictions=[float(p) for p in preds],
        )
    except NotImplementedError as e:
        raise HTTPException(status_code=501, detail=str(e))
    except Exception as e:
        logger.exception("Forecast failed")
        raise HTTPException(status_code=500, detail=str(e))
