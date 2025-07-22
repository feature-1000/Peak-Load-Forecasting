#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_synthetic_data():
    """
    Generate synthetic data for peak load forecasting.
    Creates dataset.csv and pingxiang_weather.csv with structures similar to the original data.
    """
    # Set random seed for reproducibility
    np.random.seed(42)

    # Parameters
    start_date = '2022-01-01'
    end_date = '2023-12-31'
    counties = ['Shangli', 'Shixiaqu', 'Luxi', 'Lianhua']
    hours = list(range(24))
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    n_days = len(date_range)

    # Generate dataset.csv (daily peak load)
    peak_load = np.random.normal(loc=500, scale=100, size=n_days)  # Simulate peak load
    df_load = pd.DataFrame({
        'date': date_range,
        'peak_load': peak_load
    })
    df_load['date'] = df_load['date'].dt.strftime('%Y-%m-%d')

    # Generate pingxiang_weather.csv (hourly weather data)
    weather_data = []
    for date in date_range:
        for county in counties:
            for hour in hours:
                row = {
                    'year': date.year,
                    'month': date.month,
                    'day': date.day,
                    'hour': hour,
                    'county_name': county,
                    'temperature': np.random.normal(loc=20, scale=5),  # Simulate temperature
                    'relative_humidity': np.random.uniform(30, 90),    # Simulate humidity
                    'precipitation_1h': np.random.exponential(scale=2), # Simulate precipitation
                    'wind_speed_2min': np.random.uniform(0, 10),
                    'dew_point_temp': np.random.normal(loc=15, scale=4),
                    'ground_temp': np.random.normal(loc=22, scale=5),
                    'water_vapor_pressure': np.random.uniform(5, 20)
                }
                weather_data.append(row)

    df_weather = pd.DataFrame(weather_data)

    # Save to data directory
    os.makedirs('data', exist_ok=True)
    df_load.to_csv('data/dataset.csv', index=False)
    df_weather.to_csv('data/pingxiang_weather.csv', index=False)
    print("Synthetic data saved to data/dataset.csv and data/pingxiang_weather.csv")

if __name__ == "__main__":
    generate_synthetic_data()