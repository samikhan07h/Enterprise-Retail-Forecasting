from sklearn.model_selection import TimeSeriesSplit
import numpy as np
from utils.metrics import evaluate

def time_series_cv(series, model_func, splits=3):
    tscv = TimeSeriesSplit(n_splits=splits)
    scores = []

    for train_idx, test_idx in tscv.split(series):
        train, test = series[train_idx], series[test_idx]
        preds = model_func(train, len(test))
        scores.append(evaluate(test, preds)["RMSE"])

    return round(np.mean(scores), 2)