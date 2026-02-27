import pandas as pd
from prophet import Prophet
import numpy as np

def train_prophet(train_df, forecast_steps):

    df = train_df.copy()
    df = df.rename(columns={
        df.columns[0]: "ds",
        df.columns[1]: "y"
    })

    df["ds"] = pd.to_datetime(df["ds"])

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False
    )

    model.fit(df)

    future = model.make_future_dataframe(periods=forecast_steps)
    forecast = model.predict(future)

    preds = forecast["yhat"].tail(forecast_steps).values

    return np.maximum(preds, 0)