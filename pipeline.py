import os
import joblib
import pandas as pd
import numpy as np

from models.sarima_model import train_sarima
from models.prophet_model import train_prophet
from models.xgb_model import train_xgb
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
    def train_test_split(self):
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

        # Cross-validation (SARIMA)
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
            self.best_model_object = None  # statistical models retrain for forecast

        # -------------------------------------------------
        # SAVE BEST MODEL (for API usage)
        # -------------------------------------------------
        if self.best_model_object is not None:
            os.makedirs("models", exist_ok=True)
            joblib.dump(self.best_model_object, "models/best_model.pkl")

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