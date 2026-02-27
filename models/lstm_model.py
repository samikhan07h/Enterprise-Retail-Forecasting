import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler


def create_sequences(data, look_back=14):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:i + look_back])
        y.append(data[i + look_back])
    return np.array(X), np.array(y)


def train_lstm(series, forecast_steps):

    series = np.array(series)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(series.reshape(-1, 1))

    look_back = 14

    X, y = create_sequences(scaled, look_back)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    model = Sequential()
    model.add(LSTM(50, activation='tanh', input_shape=(look_back, 1)))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=10, batch_size=32, verbose=0)

    preds = []
    last_seq = scaled[-look_back:].reshape(1, look_back, 1)

    for _ in range(forecast_steps):

        pred = model.predict(last_seq, verbose=0)
        preds.append(pred[0][0])

        last_seq = np.concatenate(
            (last_seq[:, 1:, :], pred.reshape(1, 1, 1)),
            axis=1
        )

    preds = scaler.inverse_transform(
        np.array(preds).reshape(-1, 1)
    )

    preds = np.maximum(preds.flatten(), 0)

    return model, preds