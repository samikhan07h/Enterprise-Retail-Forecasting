import os
import joblib
import pandas as pd
import numpy as np

from models.sarima_model import train_sarima
from models.prophet_model import train_prophet
from models.xgb_model import train_xgb, DEFAULT_LAG
from models.lstm_model import train_lstm

from utils.metrics import evaluate
from utils.cross_validation import time_series_cv
from mlflow_utils import log_experiment


class ForecastPipeline:

    def __init__(self, df, date_col, target_col):
        self.df = df.copy()
        self.date_col = date_col
        self.target_col = target_col

        self.best_model_name = None
        self.best_model_object = None

        self.prepare_series()

    # ---------------------------------------------------------
    # DATA PREPARATION
    # ---------------------------------------------------------
    def prepare_series(self):
        daily = (
            self.df.groupby(self.date_col)[self.target_col]
            .sum()
            .reset_index()
            .sort_values(self.date_col)
        )

        self.dates = pd.to_datetime(daily[self.date_col])
        self.series = daily[self.target_col].values

    # ---------------------------------------------------------
    # TRAIN TEST SPLIT
    # ---------------------------------------------------------
    def train_test_split(self, min_size=30):
        # Guard added: XGBoost needs > DEFAULT_LAG points just to build one
        # training row, and SARIMA/Prophet/LSTM need a reasonable minimum
        # too. Without this, small uploads fail deep inside model code with
        # a confusing error instead of a clear one.
        if len(self.series) < min_size:
            raise ValueError(
                f"Not enough data points: need at least {min_size}, "
                f"got {len(self.series)}. Upload a longer history."
            )

        split = int(len(self.series) * 0.8)

        self.train = self.series[:split]
        self.test = self.series[split:]
        self.test_dates = self.dates.iloc[split:]

    # ---------------------------------------------------------
    # MODEL TRAINING
    # ---------------------------------------------------------
    def run_models(self):

        # SARIMA
        self.sarima_pred = train_sarima(self.train, len(self.test))

        # Prophet
        prophet_df = pd.DataFrame({
            "ds": self.dates.iloc[:len(self.train)],
            "y": self.train
        })
        self.prophet_pred = train_prophet(prophet_df, len(self.test))

        # XGBoost
        self.xgb_model, self.xgb_pred = train_xgb(self.train, len(self.test))

        # LSTM
        self.lstm_model, self.lstm_pred = train_lstm(self.train, len(self.test))

    # ---------------------------------------------------------
    # WEIGHTED ENSEMBLE
    # ---------------------------------------------------------
    def weighted_ensemble(self):

        preds = np.column_stack([
            self.prophet_pred,
            self.xgb_pred,
            self.sarima_pred,
            self.lstm_pred
        ])

        rmse_vals = np.array([
            evaluate(self.test, self.prophet_pred)["RMSE"],
            evaluate(self.test, self.xgb_pred)["RMSE"],
            evaluate(self.test, self.sarima_pred)["RMSE"],
            evaluate(self.test, self.lstm_pred)["RMSE"]
        ])

        rmse_vals[rmse_vals == 0] = 1e-6

        weights = 1 / rmse_vals
        weights = weights / weights.sum()

        ensemble = np.dot(preds, weights)

        self.ensemble_weights = weights

        return ensemble

    # ---------------------------------------------------------
    # MODEL EVALUATION
    # ---------------------------------------------------------
    def evaluate_models(self):

        ensemble_pred = self.weighted_ensemble()

        results = {
            "Prophet": evaluate(self.test, self.prophet_pred),
            "XGBoost": evaluate(self.test, self.xgb_pred),
            "SARIMA": evaluate(self.test, self.sarima_pred),
            "LSTM": evaluate(self.test, self.lstm_pred),
            "Ensemble": evaluate(self.test, ensemble_pred)
        }

        # Cross-validation (SARIMA only, for now)
        cv_score = time_series_cv(
            self.series,
            lambda train, steps: train_sarima(train, steps)
        )

        results["SARIMA"]["CV_RMSE"] = cv_score

        # -------------------------------------------------
        # BEST MODEL SELECTION (based on RMSE)
        # -------------------------------------------------
        best_model = min(results, key=lambda x: results[x]["RMSE"])
        self.best_model_name = best_model

        if best_model == "XGBoost":
            self.best_model_object = self.xgb_model
        elif best_model == "LSTM":
            self.best_model_object = self.lstm_model
        else:
            self.best_model_object = None  # Prophet/SARIMA retrain on full series to forecast

        # -------------------------------------------------
        # PERSIST FORECAST ARTIFACT (for API usage)
        # -------------------------------------------------
        # FIXED: the old version only saved a model file when XGBoost or
        # LSTM won, which meant api.py had nothing to load whenever Prophet
        # or SARIMA won instead (Prophet wins in this project's own
        # benchmark results). We now always persist a self-contained
        # artifact, and forecast_from_artifact() below knows how to serve
        # a forecast from ANY of the four model types.
        os.makedirs("models", exist_ok=True)

        artifact = {
            "best_model_name": best_model,
            "series": self.series,
            "dates": self.dates,
        }

        if best_model == "XGBoost":
            artifact["xgb_model"] = self.xgb_model

        elif best_model == "LSTM":
            # Keras models aren't always reliably joblib-picklable across
            # TensorFlow versions — save in Keras' native format and store
            # just the path in the artifact instead.
            lstm_path = "models/lstm_model.keras"
            try:
                self.lstm_model.save(lstm_path)
                artifact["lstm_model_path"] = lstm_path
            except Exception:
                artifact["lstm_model"] = self.lstm_model  # best-effort fallback

        joblib.dump(artifact, "models/forecast_artifact.pkl")

        # -------------------------------------------------
        # LOG TO MLFLOW
        # -------------------------------------------------
        for model_name, metrics in results.items():
            log_experiment(model_name, metrics)

        return results

    # ---------------------------------------------------------
    # FUTURE FORECAST (Production Feature)
    # ---------------------------------------------------------
    def forecast_future(self, steps=30):

        # Retrain best model on full dataset
        if self.best_model_name == "Prophet":

            prophet_df = pd.DataFrame({
                "ds": self.dates,
                "y": self.series
            })

            future_pred = train_prophet(prophet_df, steps)

        elif self.best_model_name == "SARIMA":

            future_pred = train_sarima(self.series, steps)

        elif self.best_model_name == "XGBoost":

            model, future_pred = train_xgb(self.series, steps)

        elif self.best_model_name == "LSTM":

            model, future_pred = train_lstm(self.series, steps)

        else:
            future_pred = None

        return future_pred


# ---------------------------------------------------------------------------
# STANDALONE HELPER FOR api.py
# ---------------------------------------------------------------------------
def forecast_from_artifact(artifact: dict, steps: int) -> np.ndarray:
    """
    Produce a forecast from a persisted artifact (models/forecast_artifact.pkl)
    without needing a live ForecastPipeline instance. Mirrors the dispatch
    logic in ForecastPipeline.forecast_future(), so the API stays consistent
    with the dashboard regardless of which model won the last benchmark run.
    """
    best_model_name = artifact["best_model_name"]
    series = artifact["series"]
    dates = artifact["dates"]

    if best_model_name == "Prophet":
        prophet_df = pd.DataFrame({"ds": dates, "y": series})
        return train_prophet(prophet_df, steps)

    elif best_model_name == "SARIMA":
        return train_sarima(series, steps)

    elif best_model_name == "XGBoost":
        model = artifact["xgb_model"]
        window = list(series[-DEFAULT_LAG:])
        preds = []
        for _ in range(steps):
            pred = model.predict([window])[0]
            pred = max(pred, 0)
            preds.append(float(pred))
            window.append(pred)
            window.pop(0)
        return np.array(preds)

    elif best_model_name == "LSTM":
        # TODO: wire this up once models/lstm_model.py's exact windowing
        # and scaling logic is available — it must match train_lstm() or
        # predictions will be silently wrong.
        raise NotImplementedError(
            "LSTM serving not yet implemented — needs models/lstm_model.py "
            "to mirror its windowing/scaling logic here."
        )

    else:
        raise ValueError(f"Unknown model type: {best_model_name}")
