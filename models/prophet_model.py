import logging
import pandas as pd
import numpy as np
from prophet import Prophet

# Prophet/cmdstanpy are extremely verbose by default and will flood
# Streamlit Cloud's logs with INFO-level Stan optimizer output on every
# run. Silence it down to warnings only.
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)


def train_prophet(train_df, forecast_steps):

    df = train_df.copy()
    df = df.rename(columns={
        df.columns[0]: "ds",
        df.columns[1]: "y"
    })

    df["ds"] = pd.to_datetime(df["ds"])

    # Yearly seasonality needs at least a full year of data to estimate
    # reliably. With the pipeline's current 30-point minimum, fitting it
    # anyway just adds noisy, poorly-identified parameters — so only
    # enable it once there's enough history to actually support it.
    enough_for_yearly = len(df) >= 365

    model = Prophet(
        yearly_seasonality=enough_for_yearly,
        weekly_seasonality=True,
        daily_seasonality=False
    )

    model.fit(df)

    future = model.make_future_dataframe(periods=forecast_steps)
    forecast = model.predict(future)

    preds = forecast["yhat"].tail(forecast_steps).values

    return np.maximum(preds, 0)
