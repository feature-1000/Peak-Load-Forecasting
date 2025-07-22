# An Explainable Multi-Scale Peak Load Forecasting Model

This repository contains the implementation of the paper *"An Explainable Multi-Scale Peak Load Forecasting Model Based on Meteorological Feature Embedding and Tabular Prior-Data Fitted Networks"*. The model forecasts peak electricity load using meteorological features and employs TabPFN (Tabular Prior-Data Fitted Networks) for prediction, with SHAP analysis for interpretability.

## Project Overview

The model processes historical load and weather data, performs feature engineering (including time and meteorological features), trains multiple machine learning models, and provides visualizations and SHAP-based interpretability. Due to data privacy, the original datasets (`dataset.csv` and `pingxiang_weather.csv`) are not included. Instead, a script is provided to generate synthetic data with a similar structure for testing purposes.

### Features
- **Data Preprocessing**: Loads and preprocesses load and weather data.
- **Feature Engineering**: Creates time-based and meteorological features, including peak-hour flattening and lag features.
- **Model Training**: Trains multiple models (TabPFN, RandomForest, XGBoost, etc.) and evaluates their performance.
- **Visualization**: Generates time-series plots, scatter plots, and SHAP visualizations.
- **Interval Prediction**: Provides probabilistic predictions with confidence intervals using TabPFN.
- **SHAP Analysis**: Analyzes feature importance by location, hour, and meteorological variable.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/<your-username>/peak_load_forecasting.git
   cd peak_load_forecasting
   ```

2. **Create a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Generating Synthetic Data
Since the original data is not publicly available, you can generate synthetic data using:

```bash
python data/generate_synthetic_data.py
```

This will create `dataset.csv` and `pingxiang_weather.csv` in the `data/` directory with a structure similar to the original data.

### Running the Pipeline
1. **Preprocess Data**:
   ```bash
   python src/data_preprocessing.py
   ```
   This loads and preprocesses the data, saving it to `data/preprocessed_data.pkl`.

2. **Feature Engineering**:
   ```bash
   python src/feature_engineering.py
   ```
   This generates features and saves the processed dataset.

3. **Train Models**:
   ```bash
   python src/model_training.py
   ```
   This trains multiple models and saves evaluation results to `Results_Summary.xlsx`.

4. **Generate Visualizations**:
   ```bash
   python src/visualization.py
   ```
   This creates plots saved in the `Figure/` directory.

5. **Perform Interval Prediction**:
   ```bash
   python src/interval_prediction.py
   ```
   This generates probabilistic predictions and saves results to `tabpfn_prediction_results_with_time.csv`.

6. **Run SHAP Analysis**:
   ```bash
   python src/shap_analysis.py
   ```
   This computes SHAP values and generates visualizations saved in the `Figure/` directory.

## Project Structure

```
peak_load_forecasting/
├── data/
│   └── generate_synthetic_data.py  # Script to generate synthetic data
├── src/
│   ├── data_preprocessing.py      # Data loading and preprocessing
│   ├── feature_engineering.py     # Feature engineering and flattening
│   ├── model_training.py          # Model training and evaluation
│   ├── visualization.py           # Visualization functions
│   ├── shap_analysis.py           # SHAP analysis and visualization
│   ├── interval_prediction.py     # Interval prediction with TabPFN
├── requirements.txt               # Dependencies
├── README.md                     # Project documentation
└── LICENSE                        # License file
```

## Synthetic Data
The `generate_synthetic_data.py` script generates synthetic data with the following structure:
- **dataset.csv**: Contains `date` and `peak_load` columns.
- **pingxiang_weather.csv**: Contains weather data with columns like `year`, `month`, `day`, `hour`, `county_name`, and meteorological features (e.g., `temperature`, `relative_humidity`).

## Dependencies
See `requirements.txt` for a full list of required packages.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Citation
If you use this code in your research, please cite:

[Wei Dong]. (2025). An Explainable Multi-Scale Peak Load Forecasting Model Based on Meteorological Feature Embedding and Tabular Prior-Data Fitted Networks.

## Contact
For questions or issues, please open an issue on GitHub or contact [2023320002@nit.edu.cn].