import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error


def eval_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mse = mean_squared_error(y_true, y_pred)


    metrics = {
        "RMSE": rmse,
        "MAE": mse,
    }

    return metrics