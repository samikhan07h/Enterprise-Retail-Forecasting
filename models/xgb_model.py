import numpy as np
from xgboost import XGBRegressor


def create_features(series, lag=14):
    X, y = [], []

    for i in range(lag, len(series)):
        X.append(series[i-lag:i])
        y.append(series[i])

    return np.array(X), np.array(y)


def train_xgb(train_series, forecast_steps):

    lag = 14

    X, y = create_features(train_series, lag)

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
    last_window = train_series[-lag:].tolist()

    for _ in range(forecast_steps):

        pred = model.predict([last_window])[0]

        preds.append(max(pred, 0))

        last_window.append(pred)
        last_window.pop(0)

    return model, np.array(preds)