import numpy as np
from xgboost import XGBRegressor

DEFAULT_LAG = 14


def create_features(series, lag=DEFAULT_LAG):
    X, y = [], []

    for i in range(lag, len(series)):
        X.append(series[i - lag:i])
        y.append(series[i])

    return np.array(X), np.array(y)


def train_xgb(train_series, forecast_steps, lag=DEFAULT_LAG):

    X, y = create_features(train_series, lag)

    if len(X) == 0:
        raise ValueError(
            f"Not enough data to train XGBoost: need more than {lag} points, "
            f"got {len(train_series)}."
        )

    model = XGBRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    model.fit(X, y)

    preds = []
    last_window = list(train_series[-lag:])

    for _ in range(forecast_steps):

        pred = model.predict([last_window])[0]

        preds.append(max(pred, 0))

        last_window.append(pred)
        last_window.pop(0)

    return model, np.array(preds)
