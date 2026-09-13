import os
import joblib
import pandas as pd
import numpy as np

from models.sarima_model import train_sarima
from models.prophet_model import train_prophet
from models.xgb_model import train_xgb, DEFAULT_LAG
from models.lstm_model import train_lstm, forecast_with_lstm, load_lstm, DEFAULT_LOOK_BACK

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
        self.lstm_model, self.lstm_pred, self.lstm_scaler = train_lstm(self.train, len(self.test))

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

        # -------------------------------------------------
        # PERSIST FORECAST ARTIFACT (for API usage)
        # -------------------------------------------------
        # Two fixes from the original version:
        #
        # 1. The old code only saved a model file when XGBoost or LSTM won,
        #    so api.py had nothing to load whenever Prophet or SARIMA won
        #    instead (Prophet wins in this project's own benchmark table).
        #    We now always persist a self-contained artifact, and
        #    forecast_from_artifact() below can serve a forecast from ANY
        #    of the four model types.
        #
        # 2. The old code saved the model as trained on the 80% train
        #    split used for benchmarking. Standard practice is to use that
        #    split only to PICK the winner, then refit on the FULL series
        #    before shipping it — exactly what forecast_future() already
        #    does. We do the same here so the persisted model has seen as
        #    much data as possible.
        os.makedirs("models", exist_ok=True)

        artifact = {
            "best_model_name": best_model,
            "series": self.series,
            "dates": self.dates,
        }

        if best_model == "XGBoost":
            # forecast_steps=1 is a throwaway value here — we only need the
            # refit model, not this particular prediction.
            full_model, _ = train_xgb(self.series, forecast_steps=1)
            artifact["xgb_model"] = full_model
            self.best_model_object = full_model

        elif best_model == "LSTM":
            full_model, _, full_scaler = train_lstm(self.series, forecast_steps=1)
            lstm_path = "models/lstm_model.keras"
            scaler_path = "models/lstm_scaler.pkl"
            full_model.save(lstm_path)
            joblib.dump(full_scaler, scaler_path)
            artifact["lstm_model_path"] = lstm_path
            artifact["lstm_scaler_path"] = scaler_path
            self.best_model_object = full_model

        else:
            self.best_model_object = None  # Prophet/SARIMA retrain on demand at forecast time

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

            model, future_pred, _ = train_lstm(self.series, steps)

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
        model = load_lstm(artifact["lstm_model_path"])
        scaler = joblib.load(artifact["lstm_scaler_path"])

        scaled_series = scaler.transform(series.reshape(-1, 1))
        last_window = scaled_series[-DEFAULT_LOOK_BACK:]

        return forecast_with_lstm(model, scaler, last_window, steps, DEFAULT_LOOK_BACK)

    else:
        raise ValueError(f"Unknown model type: {best_model_name}")
