from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np

def train_sarima(train_series, forecast_steps):
    model = SARIMAX(
        train_series,
        order=(1,1,1),
        seasonal_order=(1,1,1,7),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    fitted = model.fit(disp=False)
    forecast = fitted.forecast(steps=forecast_steps)
    return np.maximum(forecast, 0)