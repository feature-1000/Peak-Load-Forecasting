#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import chinese_calendar as cn_calendar
import pickle
import os

def flatten_weather_data(df_weather, peak_hours, unique_counties, weather_features, county_mapping):
    """
    Flatten weather data for peak hours and compute statistical features.
    """
    df_weather_peak_hours = df_weather[df_weather['hour'].isin(peak_hours)]
    flattened_data = []
    for date, group in df_weather_peak_hours.groupby('date'):
        group = group.sort_values(['county_name', 'hour']).reset_index(drop=True)
        if len(group) != len(peak_hours) * len(unique_counties):
            print(f"Warning: {date} has incomplete data, skipping...")
            continue
        row_data = {'date': date}
        for feature in weather_features:
            for county in unique_counties:
                county_data = group[group['county_name'] == county]
                county_en = county_mapping.get(county, county)
                for hour in peak_hours:
                    hour_data = county_data[county_data['hour'] == hour]
                    if not hour_data.empty:
                        row_data[f'{feature}_{county}_h{hour}'] = hour_data[feature].iloc[0]
                row_data[f'{feature}_{county}_mean'] = county_data[feature].mean()
                row_data[f'{feature}_{county}_max'] = county_data[feature].max()
                row_data[f'{feature}_{county}_min'] = county_data[feature].min()
                row_data[f'{feature}_{county}_std'] = county_data[feature].std()
                row_data[f'{feature}_{county}_q25'] = county_data[feature].quantile(0.25)
                row_data[f'{feature}_{county}_q75'] = county_data[feature].quantile(0.75)
                row_data[f'{feature}_{county}_diff'] = county_data[feature].diff().mean()
        flattened_data.append(row_data)
    return pd.DataFrame(flattened_data)

def engineer_features():
    """
    Perform feature engineering and save the processed dataset.
    """
    # Load preprocessed data
    with open('data/preprocessed_data.pkl', 'rb') as f:
        data = pickle.load(f)
    df_load = data['df_load']
    df_weather = data['df_weather']

    # Parameters
    peak_hours = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
    unique_counties = df_weather['county_name'].unique()
    weather_features = ['temperature', 'relative_humidity', 'precipitation_1h', 'wind_speed_2min', 
                        'dew_point_temp', 'ground_temp', 'water_vapor_pressure']
    county_mapping = {
        '上栗县': 'Shangli',
        '市辖区': 'Shixiaqu',
        '芦溪县': 'Luxi',
        '莲花县': 'Lianhua'
    }

    # Flatten weather data
    df_weather_flattened = flatten_weather_data(df_weather, peak_hours, unique_counties, weather_features, county_mapping)
    df_weather_flattened['date'] = pd.to_datetime(df_weather_flattened['date'])

    # Merge data
    df = pd.merge(df_load, df_weather_flattened, on='date', how='inner')

    # Add time features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['weekday'] = df['date'].dt.weekday
    df['is_weekend'] = (df['weekday'] >= 5).astype(int)
    df['day_of_year'] = df['date'].dt.dayofyear
    df['quarter'] = df['date'].dt.quarter
    df['is_summer'] = ((df['month'] >= 6) & (df['month'] <= 8)).astype(int)
    df['is_winter'] = ((df['month'] >= 12) | (df['month'] <= 2)).astype(int)
    df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
    df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)
    df['sin_weekday'] = np.sin(2 * np.pi * df['weekday'] / 7)
    df['cos_weekday'] = np.cos(2 * np.pi * df['weekday'] / 7)
    df['week_of_year'] = df['date'].dt.isocalendar().week
    df['sin_day_of_year'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['cos_day_of_year'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    df['is_holiday'] = df['date'].apply(lambda x: 1 if cn_calendar.is_holiday(x) else 0)

    # Add moving averages
    df['peak_load_ma7'] = df['peak_load'].rolling(window=7).mean()
    df['peak_load_ma30'] = df['peak_load'].rolling(window=30).mean()

    # Add lag and interaction features
    for lag in range(1, 7):
        df[f'peak_load_lag{lag}'] = df['peak_load'].shift(lag)
    df['temp_summer_interaction'] = df[f'temperature_{unique_counties[0]}_mean'] * df['is_summer']
    df['heat_index'] = df[f'temperature_{unique_counties[0]}_mean'] + 0.5 * df[f'relative_humidity_{unique_counties[0]}_mean']
    df['temp_change'] = df[f'temperature_{unique_counties[0]}_mean'].diff()

    # Handle missing values
    df = df.fillna(method='ffill').fillna(method='bfill')
    df = df.fillna(df.mean())

    # Define selected features
    time_features = ['year', 'month', 'day', 'weekday', 'is_weekend', 'day_of_year', 'quarter', 'is_summer', 'is_winter', 
                     'sin_month', 'cos_month', 'sin_weekday', 'cos_weekday', 'is_holiday', 'week_of_year', 'sin_day_of_year', 'cos_day_of_year']
    weather_features = []
    for feat in weather_features:
        for county in unique_counties:
            for stat in ['mean', 'max', 'min', 'std', 'q25', 'q75', 'diff']:
                weather_features.append(f'{feat}_{county}_{stat}')
            for hour in peak_hours:
                weather_features.append(f'{feat}_{county}_h{hour}')
    other_features = [f'peak_load_lag{i}' for i in range(1, 7)] + ['temp_summer_interaction', 'heat_index', 'temp_change', 'peak_load_ma7', 'peak_load_ma30']
    selected_features = time_features + weather_features + other_features

    # Prepare training and test sets
    X = df[selected_features]
    y = df['peak_load']
    dates = df['date']
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    dates_train, dates_test = dates.iloc[:split_idx], dates.iloc[split_idx:]

    # Save processed data
    data_to_save = {
        'df': df,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'selected_features': selected_features,
        'dates_train': dates_train,
        'dates_test': dates_test
    }
    with open('data/processed_data.pkl', 'wb') as f:
        pickle.dump(data_to_save, f)
    print("Processed data saved to data/processed_data.pkl")

    return df, X_train, X_test, y_train, y_test, selected_features, dates_train, dates_test

if __name__ == "__main__":
    df, X_train, X_test, y_train, y_test, selected_features, dates_train, dates_test = engineer_features()