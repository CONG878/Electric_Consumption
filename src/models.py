from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

def train_model(ModelClass, X_train, y_train, **kwargs):
    model = ModelClass(**kwargs)
    model.fit(X_train, y_train)
    return model

def predict_model(model, X_val):
    y_pred = model.predict(X_val)
    return y_pred