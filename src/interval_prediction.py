#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from tabpfn import TabPFNRegressor
import pickle
import os

def mean_absolute_percentage_error(y_true, y_pred):
    """Calculate MAPE."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def interval_prediction():
    """
    Perform interval prediction using TabPFN and save results.
    """
    # Load processed data
    with open('data/processed_data.pkl', 'rb') as f:
        data = pickle.load(f)
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    dates_test = data['dates_test']

    # Standardize data
    x_scaler = StandardScaler()
    X_train_scaled = x_scaler.fit_transform(X_train)
    X_test_scaled = x_scaler.transform(X_test)
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).ravel()
    y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1)).ravel()

    # Fit TabPFN model
    print("\nFitting TabPFN model...")
    regressor = TabPFNRegressor(device='cuda' if torch.cuda.is_available() else 'cpu', 
                                random_state=42, n_estimators=10, softmax_temperature=0.7, memory_saving_mode='auto')
    try:
        regressor.fit(X_train_scaled, y_train_scaled)
        print("Model fitted successfully")
    except Exception as e:
        print(f"Error fitting model with scaled features: {e}")
        print("Trying with original features...")
        regressor.fit(X_train, y_train_scaled)

    # Make predictions with quantiles
    print("\nMaking predictions...")
    try:
        predictions_full = regressor.predict(X_test_scaled, output_type="full")
        predictions = predictions_full.get("predictions", predictions_full.get("mean"))
    except Exception as e:
        print(f"Error with full predictions: {e}")
        print("Falling back to standard predictions...")
        predictions = regressor.predict(X_test_scaled)

    # Evaluate
    mse_scaled = mean_squared_error(y_test_scaled, predictions)
    rmse_scaled = np.sqrt(mse_scaled)
    mae_scaled = mean_absolute_error(y_test_scaled, predictions)
    mape_scaled = mean_absolute_percentage_error(y_test_scaled, predictions)
    r2_scaled = r2_score(y_test_scaled, predictions)
    print(f"TabPFN (Scaled Data) - MSE: {mse_scaled:.4f}, RMSE: {rmse_scaled:.4f}, "
          f"MAE: {mae_scaled:.4f}, MAPE: {mape_scaled:.2f}%, R2: {r2_scaled:.4f}")

    # Calculate probability exceeding threshold
    threshold = y_train.mean()
    threshold_scaled = y_scaler.transform(np.array([[threshold]])).ravel()[0]
    quantiles = np.array(predictions_full["quantiles"])
    probabilities = np.mean(quantiles > threshold_scaled, axis=0)
    
    # Convert to original scale
    predictions_inverse = y_scaler.inverse_transform(predictions.reshape(-1, 1)).ravel()
    quantiles_original = y_scaler.inverse_transform(quantiles.T).T

    # Save results
    n_quantiles = quantiles.shape[0]
    quantile_levels = np.linspace(0.01, 0.99, n_quantiles)
    quantile_columns = [f'quantile_{q:.2f}' for q in quantile_levels]
    results_df = pd.DataFrame({
        'date': dates_test.values,
        'actual': y_test.values,
        'predicted': predictions_inverse,
        'prob_exceeds_threshold': probabilities
    })
    for i, col_name in enumerate(quantile_columns):
        results_df[col_name] = quantiles_original[i]
    
    os.makedirs('results', exist_ok=True)
    results_df.to_csv('results/tabpfn_prediction_results_with_time.csv', index=False)
    print(f"Results saved to results/tabpfn_prediction_results_with_time.csv")

    # Time-series visualization
    theme_colors = {
        'true': '#4E79A7',
        'predicted': '#E15759',
        'ci': '#76B7B2',
        'ref': '#B07AA1',
        'grid': '#D3D3D3',
        'title': '#2E2E2E',
        'label': '#4E4E4E'
    }
    
    fig1, ax1 = plt.subplots(figsize=(15, 5))
    ax1.plot(data['dates_train'], y_train, label='Training Data', color=theme_colors['true'], alpha=0.7, linewidth=1)
    ax1.plot(dates_test, y_test, label='True Values (Test)', color=theme_colors['true'], alpha=0.9, linewidth=2)
    ax1.plot(dates_test, predictions_inverse, label='Predicted Values', color=theme_colors['predicted'], alpha=0.8, linewidth=2)
    ax1.axvline(x=dates_test.iloc[0], color=theme_colors['ref'], linestyle='--', alpha=0.7, label='Train/Test Split')
    ax1.set_title('TabPFN: Complete Time Series - Peak Load Prediction', color=theme_colors['title'], fontsize=14)
    ax1.set_xlabel('Date', color=theme_colors['label'], fontsize=12)
    ax1.set_ylabel('Peak Load', color=theme_colors['label'], fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, color=theme_colors['grid'], linestyle=':', alpha=0.7)
    plt.tight_layout()
    os.makedirs('Figure', exist_ok=True)
    plt.savefig('Figure/fig1_peak_load_full_timeseries.png', dpi=300, bbox_inches='tight')
    plt.close()

    fig2, ax2 = plt.subplots(figsize=(12, 6))
    ax2.plot(dates_test, y_test, label='True Values', marker='o', linestyle='-', color=theme_colors['true'], alpha=0.8, markersize=4)
    ax2.plot(dates_test, predictions_inverse, label='Predicted Values', marker='x', linestyle='--', color=theme_colors['predicted'], alpha=0.8, markersize=4)
    ci_80_lower_idx = int(0.1 * (n_quantiles - 1))
    ci_80_upper_idx = int(0.9 * (n_quantiles - 1))
    lower_bound = quantiles_original[ci_80_lower_idx]
    upper_bound = quantiles_original[ci_80_upper_idx]
    ax2.fill_between(dates_test, lower_bound, upper_bound, color=theme_colors['ci'], alpha=0.2, label='80% Confidence Interval')
    true_outside_lower = y_test < lower_bound
    true_outside_upper = y_test > upper_bound
    true_outside_ci = true_outside_lower | true_outside_upper
    ax2.scatter(dates_test[true_outside_ci], y_test[true_outside_ci],
                color='red', s=60, marker='D', facecolors='none', edgecolors='red', 
                linewidth=1.5, label='True Values Outside CI')
    ax2.set_xlabel('Date', color=theme_colors['label'], fontsize=12)
    ax2.set_ylabel('Peak Load', color=theme_colors['label'], fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, color=theme_colors['grid'], linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.savefig('Figure/fig2_peak_load_test_period.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Histogram of predictions
    plt.figure(figsize=(10, 6))
    sns.histplot(predictions_inverse, bins=25, kde=True, stat='density', 
                 color=theme_colors['hist'], alpha=0.7, edgecolor=theme_colors['edge'])
    plt.savefig('Figure/fig_predictions_histogram.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    interval_prediction()