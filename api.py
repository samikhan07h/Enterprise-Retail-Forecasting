"""
Retail Forecasting API
-----------------------
FastAPI service that serves predictions from the best-performing model
trained by pipeline.py. Designed to run standalone or alongside the
Streamlit dashboard (e.g. for integration into other systems).

IMPORTANT: build_features() below MUST mirror the exact feature
engineering used at training time in models/xgb_model.py's
create_features(). This is currently a placeholder — replace it once
that function is available, or predictions will be silently wrong.
"""

import os
import logging
from datetime import datetime
from typing import List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("retail-forecasting-api")

MODEL_PATH = os.environ.get("MODEL_PATH", "models/best_model.pkl")

app = FastAPI(
    title="Retail Forecasting API",
    description="Serves demand forecasts from the trained best model.",
    version="1.0.0",
)

# Tighten allow_origins to your actual frontend domain(s) in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = None


def get_model():
    """Lazily load and cache the model so it's only read from disk once."""
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise HTTPException(
                status_code=503,
                detail=f"Model file not found at '{MODEL_PATH}'. Train and save a model first.",
            )
        _model = joblib.load(MODEL_PATH)
    return _model


class SalesRecord(BaseModel):
    date: datetime
    sales: float


class ForecastRequest(BaseModel):
    history: List[SalesRecord] = Field(
        ..., min_length=8,
        description="Recent historical (date, sales) records, oldest first. "
                    "Needs at least 8 records to compute lag/rolling features.",
    )
    horizon: int = Field(1, ge=1, le=90, description="Periods to forecast ahead.")


class ForecastResponse(BaseModel):
    predictions: List[float]
    model_version: str = "1.0.0"


def build_features(history: pd.DataFrame) -> pd.DataFrame:
    """
    Placeholder feature engineering.

    TODO: Replace this with the exact logic from
    models/xgb_model.py -> create_features(), so training and
    serving stay consistent. Mismatched features here silently
    produce garbage predictions instead of an error.
    """
    df = history.copy().sort_values("date")
    df["lag_1"] = df["sales"].shift(1)
    df["lag_7"] = df["sales"].shift(7)
    df["rolling_mean_7"] = df["sales"].rolling(7).mean()
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    return df.dropna()


@app.get("/")
def root():
    return {"message": "Retail Forecasting API is running", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": os.path.exists(MODEL_PATH)}


@app.post("/predict", response_model=ForecastResponse)
def predict(request: ForecastRequest):
    model = get_model()

    try:
        history_df = pd.DataFrame([r.dict() for r in request.history])
        features = build_features(history_df)

        if features.empty:
            raise HTTPException(
                status_code=400,
                detail="Not enough historical data to build features.",
            )

        X = features.drop(columns=["date", "sales"])
        prediction = model.predict(X.tail(request.horizon))
        return ForecastResponse(predictions=prediction.tolist())

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(e))
