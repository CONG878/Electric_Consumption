from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

def add_time_features(df, datetime_col='일시', drop_cols=True):
    df['datetime'] = pd.to_datetime(df[datetime_col], format='%Y%m%d %H')
    base_date = pd.to_datetime('20240601 00', format='%Y%m%d %H')
    
    df['누적일'] = (df['datetime'] - base_date).dt.days + 1.0
    df['dd^2'] = (df['누적일'] - 75.0) ** 2
    df['cos(dd)'] = np.cos(2.0 * np.pi * (df['누적일'] + 1.90) / 7.0)
    df['sin(dd)'] = np.sin(2.0 * np.pi * (df['누적일'] + 1.90) / 7.0)
    
    df['시각'] = df['datetime'].dt.hour
    df['cos(t)'] = np.cos(2.0 * np.pi * (df['시각'] - 14.45) / 24.0)
    df['sin(t)'] = np.sin(2.0 * np.pi * (df['시각'] - 14.45) / 24.0)

    if drop_cols:
        cols_to_drop = [datetime_col, 'datetime', '시각']
        df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

    return df

def drop_unused_features(df, columns=['건물번호', '냉방면적(m2)', 'num_date_time']):
    return df.drop(columns=[col for col in columns if col in df.columns])

def split_features_target(df, target='전력소비량(kWh)', drop_cols=None):
    if drop_cols is None:
        drop_cols = ['기온(°C)', '강수량(mm)', '풍속(m/s)', '습도(%)', '일조(hr)', '일사(MJ/m2)', 'PCS용량(kW)', 'ESS저장용량(kWh)']
    X = df.drop(columns=[target] + drop_cols)
    y = df[target]
    return X, y

def train_valid_split(X_scaled, y, test_size=0.2, random_state=42):
    return train_test_split(X_scaled, y, test_size=test_size, random_state=random_state)

def scale_features(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

def fit_weather_pca(df, weather_cols=['기온(°C)', '강수량(mm)', '풍속(m/s)', '습도(%)']):
    """
    학습 데이터로부터 Scaler와 PCA를 fit 하고 반환합니다.
    """
    weather_scaled, scaler = scale_features(df[weather_cols])

    pca = PCA(n_components=1, random_state=42)
    pca.fit(weather_scaled)

    return scaler, pca

def transform_weather_pca(df, scaler, pca, weather_cols=['기온(°C)', '강수량(mm)', '풍속(m/s)', '습도(%)']):
    """
    fit된 Scaler와 PCA를 이용해 주성분을 계산하여 df에 추가합니다.
    """
    weather_scaled = scaler.transform(df[weather_cols])
    
    pc = pca.transform(weather_scaled)

    df = df.copy()
    df['weather_PC1'] = pc
    return df
