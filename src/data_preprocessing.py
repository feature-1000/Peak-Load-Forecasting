#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import pickle

def load_and_preprocess_data(load_file='data/dataset.csv', weather_file='data/pingxiang_weather.csv'):
    """
    Load and preprocess load and weather data.
    """
    # County name mapping
    county_mapping = {
        '上栗县': 'Shangli',
        '市辖区': 'Shixiaqu',
        '芦溪县': 'Luxi',
        '莲花县': 'Lianhua'
    }

    # Load data
    df_load = pd.read_csv(load_file)
    df_weather = pd.read_csv(weather_file)

    # Apply county name mapping
    df_weather['county_name'] = df_weather['county_name'].map(county_mapping)

    # Convert date and datetime
    df_load['date'] = pd.to_datetime(df_load['date'])
    df_weather['datetime'] = pd.to_datetime(df_weather[['year', 'month', 'day', 'hour']])
    df_weather['date'] = df_weather['datetime'].dt.date
    df_weather = df_weather.sort_values(['date', 'county_name', 'hour']).reset_index(drop=True)

    # Save preprocessed data
    os.makedirs('data', exist_ok=True)
    with open('data/preprocessed_data.pkl', 'wb') as f:
        pickle.dump({'df_load': df_load, 'df_weather': df_weather}, f)
    print("Preprocessed data saved to data/preprocessed_data.pkl")

    return df_load, df_weather

if __name__ == "__main__":
    df_load, df_weather = load_and_preprocess_data()