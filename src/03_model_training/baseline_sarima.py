import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import os
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    mean_absolute_percentage_error,
)
from sklearn.preprocessing import MinMaxScaler

# 1. Setup MLflow Experiment
mlflow.set_experiment("MTA_Ridership_Forecasting")
DATA_PATH = "data/gold/mta_sarima.parquet"


def log_model_run(run_name, model_type, params, y_true, y_pred, results_obj=None):
    """Logs parameters, metrics, forecasts, diagnostics, and custom residual plots."""
    with mlflow.start_run(run_name=run_name):
        latest_data_date = df.index.max().strftime("%Y-%m-%d")
        total_days = len(df)

        # Set tags for better organization and filtering in MLflow UI
        mlflow.set_tag("data_version_date", latest_data_date)
        mlflow.set_tag("dataset_row_count", total_days)
        mlflow.set_tag("project_phase", "EDA_Incremental_Update")

        # --- DATA REPRODUCIBILITY ---
        mlflow.log_artifact(DATA_PATH)
        mlflow.log_param("model_type", model_type)
        mlflow.log_params(params)

        # Calculate Metrics
        mae = mean_absolute_error(y_true, y_pred)
        rmse = root_mean_squared_error(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred)

        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mape", mape)

        if results_obj is not None:
            mlflow.log_metric("aic", results_obj.aic)

        # --- PLOT 1: Forecast Comparison ---
        plt.figure(figsize=(10, 5))
        plt.plot(y_true.index, y_true, label="Actual", color="black", alpha=0.6)
        plt.plot(y_true.index, y_pred, label="Predicted", linestyle="--", color="red")
        plt.title(f"{run_name}\nMAE: {mae:.3f}")
        plt.legend()

        forecast_path = f"reports/plots/{run_name}_forecast.png"
        plt.savefig(forecast_path)
        mlflow.log_artifact(forecast_path)
        plt.close()

        # --- PLOT 2: Targeted Residual Plot (The "Check") ---
        # This shows if the errors are random or have a pattern
        residuals = y_true - y_pred
        plt.figure(figsize=(8, 5))
        plt.scatter(y_pred, residuals, alpha=0.5, color="purple")
        plt.axhline(y=0, color="r", linestyle="--")
        plt.title(f"Residuals vs Predicted: {run_name}")
        plt.xlabel("Predicted Values")
        plt.ylabel("Residual Error")

        res_path = f"reports/plots/{run_name}_residuals.png"
        plt.savefig(res_path)
        mlflow.log_artifact(res_path)
        plt.close()

        # --- PLOT 3: Statsmodels Diagnostics (Standard tool) ---
        if results_obj:
            results_obj.plot_diagnostics(figsize=(12, 8))
            diag_path = f"reports/plots/{run_name}_diagnostics.png"
            plt.savefig(diag_path)
            mlflow.log_artifact(diag_path)
            plt.close()

        print(f"✅ Finished Logging: {run_name}")


# 2. Data Preparation
os.makedirs("reports/plots", exist_ok=True)
df = pd.read_parquet(DATA_PATH)
df.index = pd.to_datetime(df.index)
df = df.asfreq("D")
y = df["daily_ridership"] / 1_000_000  # Scaling to Millions for readability

split_date = y.index[-30]
train_y, test_y = y.loc[: split_date - pd.Timedelta(days=1)], y.loc[split_date:]
forecast_steps = len(test_y)

# Feature selection for SARIMAX
exog_cols = ["temp", "precip", "snow_lag1", "is_holiday"]
exog_raw = df[exog_cols]
train_exog_raw = exog_raw.loc[: split_date - pd.Timedelta(days=1)]
test_exog_raw = exog_raw.loc[split_date:]

# Scaling Exogenous features (avoids data leakage)
scaler = MinMaxScaler()
train_exog = pd.DataFrame(
    scaler.fit_transform(train_exog_raw), index=train_y.index, columns=exog_cols
)
test_exog = pd.DataFrame(
    scaler.transform(test_exog_raw), index=test_y.index, columns=exog_cols
)

# Common SARIMA settings
sarima_config = {"enforce_stationarity": False, "enforce_invertibility": False}
sarima_params = {"order": (1, 1, 1), "seasonal_order": (1, 1, 1, 7)}

# --- EXECUTION ---

# 1. Seasonal Naive (The baseline to beat)
sn_pred = np.tile(train_y.tail(7).values, int(np.ceil(forecast_steps / 7)))[
    :forecast_steps
]
log_model_run(
    "Baseline_01_Seasonal_Naive",
    "Naive",
    {"period": 7},
    test_y,
    pd.Series(sn_pred, index=test_y.index),
)

# 2. Simple SARIMA (No weather data)
model_sarima = SARIMAX(train_y, **sarima_params, **sarima_config).fit(disp=False)
log_model_run(
    "Baseline_02_Simple_SARIMA",
    "SARIMA",
    sarima_params,
    test_y,
    model_sarima.get_forecast(steps=forecast_steps).predicted_mean,
    model_sarima,
)

# 3. Controlled SARIMAX (Winner from previous runs)
model_controlled = SARIMAX(
    train_y, exog=train_exog, **sarima_params, **sarima_config
).fit(disp=False)
log_model_run(
    "Model_02.5_SARIMAX_Controlled",
    "SARIMAX",
    sarima_params,
    test_y,
    model_controlled.get_forecast(steps=forecast_steps, exog=test_exog).predicted_mean,
    model_controlled,
)

# 4. Tuned SARIMAX (Your experimental run)
tuned_params = {"order": (0, 1, 0), "seasonal_order": (1, 0, 1, 7)}
model_tuned = SARIMAX(train_y, exog=train_exog, **tuned_params, **sarima_config).fit(
    disp=False
)
log_model_run(
    "Model_03_SARIMAX_Tuned",
    "SARIMAX",
    tuned_params,
    test_y,
    model_tuned.get_forecast(steps=forecast_steps, exog=test_exog).predicted_mean,
    model_tuned,
)

print("\n All models logged.")
