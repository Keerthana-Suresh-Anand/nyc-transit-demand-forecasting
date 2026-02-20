import pandas as pd
import mlflow
import pmdarima as pm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# 1. Setup
mlflow.set_experiment("MTA_Ridership_Forecasting")
DATA_PATH = "data/gold/mta_sarima.parquet"

# 2. Load & Prepare Data (Leakage-Free)
df = pd.read_parquet(DATA_PATH)
df.index = pd.to_datetime(df.index)
df = df.asfreq("D")
y = df["daily_ridership"] / 1_000_000

split_date = y.index[-30]
train_y, test_y = y.loc[: split_date - pd.Timedelta(days=1)], y.loc[split_date:]

exog_cols = ["temp", "precip", "snow_lag1", "is_holiday"]
scaler = MinMaxScaler()
# Fit ONLY on train
train_exog = pd.DataFrame(
    scaler.fit_transform(df.loc[train_y.index, exog_cols]),
    index=train_y.index,
    columns=exog_cols,
)
test_exog = pd.DataFrame(
    scaler.transform(df.loc[test_y.index, exog_cols]),
    index=test_y.index,
    columns=exog_cols,
)

# 3. Auto-ARIMA Search
print("🚀 Starting Auto-ARIMA Stepwise Search...")

# We use the training data to find the best model structure
auto_model = pm.auto_arima(
    train_y,
    exog=train_exog,
    start_p=0,
    start_q=0,
    max_p=3,
    max_q=3,
    m=7,  # Weekly seasonality
    seasonal=True,
    start_P=0,
    D=1,  # Force seasonal differencing for stability
    trace=True,  # Shows the search progress
    error_action="ignore",
    suppress_warnings=True,
    stepwise=True,  # Efficient search algorithm
)

best_order = auto_model.order
best_seasonal = auto_model.seasonal_order

print(f"\n✅ Best Model Found: SARIMAX{best_order}x{best_seasonal}")

# 4. Log the "Champion" to MLflow
with mlflow.start_run(run_name="Model_04_AutoARIMA_Champion"):
    # Re-fit the best model to get the full results object for diagnostics
    final_model = SARIMAX(
        train_y,
        exog=train_exog,
        order=best_order,
        seasonal_order=best_seasonal,
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)

    y_pred = final_model.get_forecast(steps=len(test_y), exog=test_exog).predicted_mean

    # Metrics
    mae = mean_absolute_error(test_y, y_pred)
    rmse = root_mean_squared_error(test_y, y_pred)

    # Logging
    mlflow.log_artifact(DATA_PATH)
    mlflow.log_params(
        {
            "order": best_order,
            "seasonal_order": best_seasonal,
            "search_method": "stepwise_auto_arima",
        }
    )
    mlflow.log_metrics({"mae": mae, "rmse": rmse, "aic": final_model.aic})

    # Forecast Plot
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.plot(test_y.index, test_y, label="Actual")
    plt.plot(test_y.index, y_pred, label="Auto-ARIMA", linestyle="--")
    plt.title(f"Auto-ARIMA Champion | MAE: {mae:.3f}")
    plt.legend()
    plt.savefig("reports/plots/auto_arima_forecast.png")
    mlflow.log_artifact("reports/plots/auto_arima_forecast.png")

    print(f"Final Champion MAE: {mae:.4f}")
