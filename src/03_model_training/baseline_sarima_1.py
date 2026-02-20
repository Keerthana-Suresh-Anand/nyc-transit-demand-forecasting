import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import os

# 1. Load Gold Data
df = pd.read_parquet("data/gold/mta_sarima.parquet")
df.index = pd.to_datetime(df.index)
df = df.asfreq("D")

# 2. Prepare Series
# Simple scaling: Predict in "Millions" to help the math converge
y = df["daily_ridership"] / 1_000_000

# 3. Train/Test Split (Time-based)
# Use the last 30 days as a "Holdout" set to test the model
split_date = y.index[-30]
train = y.loc[: split_date - pd.Timedelta(days=1)]
test = y.loc[split_date:]

# 4. Define Baseline SARIMA Parameters
# (p,d,q) x (P,D,Q,s)
# Starting with (1,1,1) x (1,1,1,7) as a robust baseline
model = SARIMAX(
    train,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 7),
    enforce_stationarity=False,
    enforce_invertibility=False,
)

results = model.fit(disp=False)

# 5. Forecast
forecast_steps = 30
forecast = results.get_forecast(steps=forecast_steps)
y_pred = forecast.predicted_mean
conf_int = forecast.conf_int()

os.makedirs("reports/plots", exist_ok=True)

results.plot_diagnostics(figsize=(12, 8))
plt.savefig("reports/plots/baseline_forecast.png")
plt.close()

# 6. Evaluate
mae = mean_absolute_error(test, y_pred)
rmse = root_mean_squared_error(test, y_pred)

print(f"--- Baseline Performance ---")
print(f"MAE: {mae:.4f} million riders")
print(f"RMSE: {rmse:.4f} million riders")

print(results.summary())

# 7. Quick Plot
plt.figure(figsize=(12, 6))
plt.plot(train.index[-60:], train.tail(60), label="Past Data")
plt.plot(test.index, test, label="Actual (Test)")
plt.plot(test.index, y_pred, label="Forecast", linestyle="--")
plt.fill_between(
    test.index,
    conf_int.iloc[:, 0],
    conf_int.iloc[:, 1],
    color="gray",
    alpha=0.3,
    label="Confidence Interval",
)
plt.title("Baseline SARIMA: Ridership Forecast")
plt.legend()
plt.savefig("reports/plots/baseline_forecast.png")
plt.close()
