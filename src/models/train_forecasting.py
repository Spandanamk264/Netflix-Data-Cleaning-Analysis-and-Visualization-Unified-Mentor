
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_squared_error
from loguru import logger
import matplotlib.pyplot as plt
import joblib
import os

logger.add("logs/forecasting.log")
MODEL_DIR = "models/forecasting"
os.makedirs(MODEL_DIR, exist_ok=True)

def train_forecaster():
    logger.info("Loading Data for Forecasting...")
    df = pd.read_csv(r"data/processed/netflix_features.csv")
    
    # Preprocess for Prophet
    # We need 'ds' (date) and 'y' (value)
    # Task: Predict monthly volume of new titles
    
    if 'date_added_clean' not in df.columns:
        # Re-parse if lost in CSV roundtrip
        # Actually features csv has 'add_month' etc but maybe not raw date.
        # Let's load processed clean v2 for full date
        df_raw = pd.read_csv(r"data/processed/netflix_cleaned_v2.csv")
        df_raw['date_added_clean'] = pd.to_datetime(df_raw['date_added_clean'])
        df = df_raw.copy()
    
    # Drop NaNs
    df = df.dropna(subset=['date_added_clean'])
    df['date_added_clean'] = pd.to_datetime(df['date_added_clean'])
    
    # Aggregation
    df_ts = df.set_index('date_added_clean').resample('M').size().reset_index(name='y')
    df_ts.columns = ['ds', 'y']
    
    # Split
    train_size = int(len(df_ts) * 0.8)
    train = df_ts.iloc[:train_size]
    test = df_ts.iloc[train_size:]
    
    logger.info(f"Training Prophet on {len(train)} months, Testing on {len(test)} months")
    
    # Model
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    m.add_country_holidays(country_name='US') 
    m.fit(train)
    
    # Predict
    future = m.make_future_dataframe(periods=len(test), freq='M')
    forecast = m.predict(future)
    
    # Evaluate
    preds = forecast.iloc[-len(test):]['yhat'].values
    rmse = np.sqrt(mean_squared_error(test['y'], preds))
    logger.info(f"Prophet Forecast RMSE: {rmse:.4f}")
    
    # Save
    joblib.dump(m, f"{MODEL_DIR}/prophet_model.joblib")
    forecast.to_csv(f"{MODEL_DIR}/forecast_results.csv")
    
    logger.success("Forecasting Complete")

if __name__ == "__main__":
    train_forecaster()
