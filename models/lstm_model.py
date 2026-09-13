import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

DEFAULT_LOOK_BACK = 14


def create_sequences(data, look_back=DEFAULT_LOOK_BACK):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:i + look_back])
        y.append(data[i + look_back])
    return np.array(X), np.array(y)


def forecast_with_lstm(model, scaler, last_scaled_window, forecast_steps, look_back=DEFAULT_LOOK_BACK):
    """
    Recursively forecast forward using an already-trained model + scaler.
    Extracted out of train_lstm() so both post-training evaluation and
    API serving (forecast_from_artifact, no retraining) share the exact
    same recursive-prediction logic instead of duplicating it.
    """
    last_seq = np.array(last_scaled_window).reshape(1, look_back, 1)
    preds = []

    for _ in range(forecast_steps):
        pred = model.predict(last_seq, verbose=0)
        preds.append(pred[0][0])
        last_seq = np.concatenate(
            (last_seq[:, 1:, :], pred.reshape(1, 1, 1)),
            axis=1
        )

    preds = scaler.inverse_transform(np.array(preds).reshape(-1, 1))
    return np.maximum(preds.flatten(), 0)


def train_lstm(series, forecast_steps, look_back=DEFAULT_LOOK_BACK):

    series = np.array(series)

    if len(series) <= look_back:
        raise ValueError(
            f"Not enough data to train LSTM: need more than {look_back} points, "
            f"got {len(series)}."
        )

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(series.reshape(-1, 1))

    X, y = create_sequences(scaled, look_back)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    model = Sequential()
    model.add(LSTM(50, activation='tanh', input_shape=(look_back, 1)))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=10, batch_size=32, verbose=0)

    last_window = scaled[-look_back:]
    preds = forecast_with_lstm(model, scaler, last_window, forecast_steps, look_back)

    return model, preds, scaler


def load_lstm(path):
    """Small wrapper so callers don't need to import Keras directly."""
    return load_model(path)
