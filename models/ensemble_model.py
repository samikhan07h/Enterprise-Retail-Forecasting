import numpy as np

def weighted_ensemble(predictions_dict, rmse_dict):

    model_names = list(predictions_dict.keys())

    preds_matrix = np.column_stack([
        predictions_dict[name] for name in model_names
    ])

    rmse_vals = np.array([
        rmse_dict[name] for name in model_names
    ])

    rmse_vals[rmse_vals == 0] = 1e-6

    weights = 1 / rmse_vals
    weights = weights / weights.sum()

    ensemble_pred = np.dot(preds_matrix, weights)

    return ensemble_pred, dict(zip(model_names, weights))