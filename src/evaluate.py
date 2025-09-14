from sklearn.metrics import r2_score, mean_squared_error, root_mean_squared_error
import numpy as np

def SMAPE(y_true, y_pred):
    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    denominator = np.where(denominator == 0, 1e-8, denominator)
    return np.mean(numerator / denominator) * 100

def evaluate_model(name, y_val, y_pred):
    smape = SMAPE(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    mse = mean_squared_error(y_val, y_pred)
    rmse = root_mean_squared_error(y_val, y_pred)
    
    print(f"[{name}] SMAPE: {smape:.4f} | MSE: {mse:.4f} | RMSE: {rmse:.4f} | R²: {r2:.4f}")
    
    return {"name": name, "SMAPE": smape, "R2": r2, "MSE": mse, "RMSE": rmse}