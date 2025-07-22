#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from tabpfn import TabPFNRegressor
from tabpfn_extensions import interpretability
from sklearn.impute import SimpleImputer
import torch
import pickle
import os

def shap_analysis():
    """
    Perform SHAP analysis and generate visualizations.
    """
    # Load processed data
    with open('data/processed_data.pkl', 'rb') as f:
        data = pickle.load(f)
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']

    # Standardize data
    x_scaler = StandardScaler()
    X_train_scaled = x_scaler.fit_transform(X_train)
    X_test_scaled = x_scaler.transform(X_test)
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).ravel()

    # Fit TabPFN model
    regressor = TabPFNRegressor(device='cuda' if torch.cuda.is_available() else 'cpu', 
                                random_state=42, n_estimators=10, softmax_temperature=0.7, memory_saving_mode='auto')
    regressor.fit(X_train_scaled, y_train_scaled)

    # Compute SHAP values
    feature_names = X_train.columns.tolist()
    X_test_data = X_test_scaled[:50]  # Limit to 50 samples
    if np.any(np.isnan(X_test_data)):
        print("Warning: NaNs detected in X_test. Imputing with mean...")
        imputer = SimpleImputer(strategy='mean')
        X_test_data = imputer.fit_transform(X_test_data)
    
    try:
        shap_values = interpretability.shap.get_shap_values(
            estimator=regressor,
            test_x=X_test_data,
            attribute_names=feature_names,
            algorithm="permutation",
            max_evals=600
        )
    except Exception as e:
        print(f"Permutation algorithm error: {e}")
        print("Trying Exact algorithm...")
        shap_values = interpretability.shap.get_shap_values(
            estimator=regressor,
            test_x=X_test_data,
            attribute_names=feature_names,
            algorithm="exact"
        )

    # Save SHAP values
    os.makedirs('shap_values', exist_ok=True)
    np.save('shap_values/shap_values.npy', shap_values.values)
    np.save('shap_values/feature_names.npy', feature_names)
    print("SHAP values saved to shap_values/ directory")

    # Plot SHAP summary
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")
    shap.summary_plot(shap_values, feature_names=feature_names, plot_type="dot", show=False)
    plt.title("SHAP Summary Plot (Dot)", color='#2E2E2E', fontsize=12)
    plt.tight_layout()
    os.makedirs('Figure', exist_ok=True)
    plt.savefig('Figure/shap_summary_dot.png', dpi=300, bbox_inches='tight')
    plt.close()

    shap.summary_plot(shap_values, feature_names=feature_names, plot_type="bar", show=False)
    plt.title("SHAP Summary Plot (Bar)", color='#2E2E2E', fontsize=12)
    plt.tight_layout()
    plt.savefig('Figure/shap_summary_bar.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Location SHAP influence
    locations = ['Shangli', 'Shixiaqu', 'Luxi', 'Lianhua']
    location_shap_sums = {loc: 0.0 for loc in locations}
    for i, feature in enumerate(feature_names):
        for loc in locations:
            if loc.lower() in feature.lower():
                location_shap_sums[loc] += np.mean(np.abs(shap_values.values[:, i]))
    
    plt.figure(figsize=(8, 3))
    bars = plt.bar(location_shap_sums.keys(), location_shap_sums.values(), 
                   color=['#8B9DC3', '#DEB887', '#B19CD9', '#87CEEB'], alpha=0.7)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height, f'{height:.4f}', 
                 ha='center', va='bottom', fontsize=10, color='#333333')
    plt.ylabel('Sum of Absolute SHAP Values (Mean)', fontsize=6, color='#555555')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('Figure/location_shap_influence.png', dpi=300)
    plt.close()

    # Hour SHAP influence
    hours = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
    hour_shap_sums = {f'h{hour}': 0.0 for hour in hours}
    for i, feature in enumerate(feature_names):
        for hour in hours:
            if f'h{hour}' in feature.lower():
                hour_shap_sums[f'h{hour}'] += np.mean(np.abs(shap_values.values[:, i]))
    
    plt.figure(figsize=(8, 3))
    hour_names = [f'Hour {hour}' for hour in hours]
    colors = sns.color_palette("husl", len(hour_names))
    bars = plt.bar(hour_names, list(hour_shap_sums.values()), color=colors, alpha=0.7)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height, f'{height:.4f}', 
                 ha='center', va='bottom', fontsize=9, color='#333333', rotation=0)
    plt.ylabel('Sum of Absolute SHAP Values (Mean)', fontsize=6, color='#555555')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('Figure/hour_shap_influence.png', dpi=300)
    plt.close()

if __name__ == "__main__":
    shap_analysis()