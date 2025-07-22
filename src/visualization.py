#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

def plot_model_comparisons():
    """
    Plot scatter and time-series comparisons for top models.
    """
    # Load model predictions
    with open('results/model_predictions.pkl', 'rb') as f:
        data = pickle.load(f)
    results = data['results']
    dates_test = data['dates_test']
    y_test = data['y_test']

    # Theme colors
    theme_colors = {
        'true': '#4E79A7',
        'predicted': '#E15759',
        'ci': '#76B7B2',
        'ref': '#B07AA1',
        'hist': '#59A14F',
        'title': '#2E2E2E',
        'label': '#4E4E4E',
        'grid': '#D3D3D3',
        'edge': '#333333'
    }

    # Scatter plot for all models
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")
    fig, axes = plt.subplots(2, 4, figsize=(16, 10))
    axes = axes.flatten()
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F']
    all_models = list(results.keys())
    
    for i, model_name in enumerate(all_models):
        ax = axes[i]
        y_actual = y_test.values
        y_predicted = results[model_name]['y_pred']
        ax.scatter(y_actual, y_predicted, alpha=0.6, s=40, color=colors[i % len(colors)], edgecolors='white', linewidth=0.5)
        min_val = min(y_actual.min(), y_predicted.min())
        max_val = max(y_actual.max(), y_predicted.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1, alpha=0.8)
        r2_value = results[model_name]['R2_scaled']
        ax.text(0.05, 0.95, f'R² = {r2_value:.4f}', transform=ax.transAxes, fontsize=10, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax.set_title(f'{model_name}', fontsize=8, fontweight='bold', pad=15)
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('gray')
        ax.spines['bottom'].set_color('gray')
        if i >= 4:
            ax.set_xlabel('Actual Values', fontsize=10)
        if i % 4 == 0:
            ax.set_ylabel('Predicted Values', fontsize=10)
    
    plt.subplots_adjust(hspace=-0.3, wspace=0.3)
    os.makedirs('Figure', exist_ok=True)
    plt.savefig('Figure/model_comparison_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Time-series plot for top 4 models
    top4_models = ['TabPFN', 'RandomForest', 'LightGBM', 'GradientBoosting']
    top4_styles = {
        'TabPFN': {'color': '#E15759', 'linestyle': '-', 'marker': 'X', 'markersize': 5, 'linewidth': 2},
        'RandomForest': {'color': '#59A14F', 'linestyle': '--', 'marker': 's', 'markersize': 4, 'linewidth': 1.5},
        'LightGBM': {'color': '#76B7B2', 'linestyle': '-.', 'marker': 'D', 'markersize': 4, 'linewidth': 1.5},
        'GradientBoosting': {'color': '#B07AA1', 'linestyle': ':', 'marker': '*', 'markersize': 5, 'linewidth': 1.5}
    }
    
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.plot(dates_test, y_test, label='True Values', color=theme_colors['true'], linewidth=2.5, zorder=10, alpha=0.9)
    for name in top4_models:
        style = top4_styles[name]
        ax.plot(dates_test, results[name]['y_pred'], 
                label=f'{name} (R²={results[name]["R2_scaled"]:.3f})',
                color=style['color'], linestyle=style['linestyle'],
                marker=style['marker'], markersize=style['markersize'],
                linewidth=style['linewidth'], alpha=0.85)
    
    ax.set_xlabel('Date', color=theme_colors['label'], fontsize=12, labelpad=10)
    ax.set_ylabel('Peak Load', color=theme_colors['label'], fontsize=12, labelpad=10)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.08), ncol=5, fontsize=12,
              frameon=True, fancybox=True, shadow=False, columnspacing=1.5)
    ax.grid(True, color=theme_colors['grid'], linestyle=':', alpha=0.7)
    plt.gcf().autofmt_xdate()
    plt.subplots_adjust(top=0.85)
    plt.tight_layout()
    plt.savefig('Figure/fig4_top4_models_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot for specific date range
    start_date = '2023-10-01'
    end_date = '2023-11-01'
    mask = (dates_test >= start_date) & (dates_test <= end_date)
    filtered_dates = dates_test[mask]
    filtered_y_test = y_test[mask]
    filtered_predictions = {name: results[name]['y_pred'][mask] for name in top4_models}
    
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(filtered_dates, filtered_y_test, color=theme_colors['true'], linewidth=2.5, alpha=0.9, zorder=10)
    for name in top4_models:
        style = top4_styles[name]
        ax.plot(filtered_dates, filtered_predictions[name], 
                color=style['color'], linestyle=style['linestyle'],
                marker=style['marker'], markersize=style['markersize'],
                linewidth=style['linewidth'], alpha=0.85)
    
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticklabels([])
    ax.grid(True, color=theme_colors['grid'], linestyle=':', alpha=0.7)
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.savefig('Figure/fig5_sep_oct_prediction.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Visualizations saved to Figure/ directory")

if __name__ == "__main__":
    plot_model_comparisons()