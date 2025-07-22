#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from tabpfn import TabPFNRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
import time
import psutil
import pickle
import os

def mean_absolute_percentage_error(y_true, y_pred):
    """Calculate MAPE."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def train_and_evaluate_models():
    """
    Train and evaluate multiple models, saving results.
    """
    # Load processed data
    with open('data/processed_data.pkl', 'rb') as f:
        data = pickle.load(f)
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    dates_train = data['dates_train']
    dates_test = data['dates_test']

    # Standardize data
    x_scaler = StandardScaler()
    X_train_scaled = x_scaler.fit_transform(X_train)
    X_test_scaled = x_scaler.transform(X_test)
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).ravel()
    y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1)).ravel()

    # Initialize models
    models = {
        'DecisionTree': DecisionTreeRegressor(random_state=42),
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=100, random_state=42),
        'LightGBM': LGBMRegressor(n_estimators=100, random_state=42, verbose=-1),
        'AdaBoost': make_pipeline(SimpleImputer(strategy='mean'), AdaBoostRegressor(n_estimators=100, random_state=42)),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'CatBoost': CatBoostRegressor(n_estimators=100, random_state=42, verbose=0),
        'TabPFN': TabPFNRegressor(device='cuda' if torch.cuda.is_available() else 'cpu', random_state=42, n_estimators=100, softmax_temperature=0.6, memory_saving_mode='auto')
    }

    # Train and evaluate
    results = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        start_time = time.time()
        model.fit(X_train_scaled, y_train_scaled)
        training_time = time.time() - start_time

        # Memory usage
        def get_memory_usage():
            return psutil.Process().memory_info().rss / 1024 / 1024  # MB
        mem_before = get_memory_usage()
        model.fit(X_train_scaled, y_train_scaled)
        mem_after = get_memory_usage()
        mem_peak = max(get_memory_usage() for _ in range(10))
        memory_usage = mem_peak - mem_before

        # Predict
        y_pred_scaled = model.predict(X_test_scaled)
        y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

        # Evaluate
        mse_scaled = mean_squared_error(y_test_scaled, y_pred_scaled)
        rmse_scaled = np.sqrt(mse_scaled)
        mae_scaled = mean_absolute_error(y_test_scaled, y_pred_scaled)
        mape_scaled = mean_absolute_percentage_error(y_test_scaled, y_pred_scaled)
        r2_scaled = r2_score(y_test_scaled, y_pred_scaled)

        results[name] = {
            'MSE_scaled': mse_scaled,
            'RMSE_scaled': rmse_scaled,
            'MAE_scaled': mae_scaled,
            'MAPE_scaled': mape_scaled,
            'R2_scaled': r2_scaled,
            'Training_time': training_time,
            'Memory_usage': memory_usage,
            'y_pred': y_pred
        }
        print(f"{name} (Scaled Data) - MSE: {mse_scaled:.4f}, RMSE: {rmse_scaled:.4f}, "
              f"MAE: {mae_scaled:.4f}, MAPE: {mape_scaled:.2f}%, R2: {r2_scaled:.4f}, "
              f"Training Time: {training_time:.2f}s, Memory Usage: {memory_usage:.2f}MB")

    # Save results
    results_df = pd.DataFrame({
        'Model': list(results.keys()),
        'R2': [results[name]['R2_scaled'] for name in results],
        'MSE': [results[name]['MSE_scaled'] for name in results],
        'RMSE': [results[name]['RMSE_scaled'] for name in results],
        'MAE': [results[name]['MAE_scaled'] for name in results],
        'MAPE(%)': [results[name]['MAPE_scaled'] for name in results],
        'Training Time(s)': [results[name]['Training_time'] for name in results],
        'Memory Usage(MB)': [results[name]['Memory_usage'] for name in results]
    })
    os.makedirs('results', exist_ok=True)
    results_df.to_excel('results/Results_Summary.xlsx', index=False)
    print("\nResults saved to results/Results_Summary.xlsx")

    # Save model predictions
    with open('results/model_predictions.pkl', 'wb') as f:
        pickle.dump({'results': results, 'dates_test': dates_test, 'y_test': y_test}, f)
    print("Model predictions saved to results/model_predictions.pkl")

if __name__ == "__main__":
    train_and_evaluate_models()