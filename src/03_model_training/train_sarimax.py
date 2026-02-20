import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
import warnings
import os
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
)
from sklearn.preprocessing import MinMaxScaler
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy import stats

warnings.filterwarnings("ignore")

# -----------------------------
# 1. Load and Prepare Data
# -----------------------------
df = pd.read_parquet("data/gold/mta_sarima.parquet")
df.index = pd.to_datetime(df.index)
df = df.asfreq("D")

y = df["daily_ridership"] / 1_000_000
exog_cols = ["temp", "precip", "snow_lag1", "is_holiday"]

scaler = MinMaxScaler()
df[exog_cols] = scaler.fit_transform(df[exog_cols])
exog = df[exog_cols]

# -----------------------------
# 2. Define Parameter Grid
# -----------------------------
p = d = q = range(0, 2)  # you can expand later to 0-3 if you want
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 7) for x in pdq]

# -----------------------------
# 3. Hyperparameter Tuning (Grid Search)
# -----------------------------
print("--- Starting Hyperparameter Tuning ---")
best_mae = float("inf")
best_params = None

tune_y = y.iloc[:-30]
tune_exog = exog.iloc[:-30]

for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            model = SARIMAX(
                tune_y,
                exog=tune_exog,
                order=param,
                seasonal_order=param_seasonal,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            results = model.fit(disp=False)
            # Evaluate using last 14 days for a slightly more stable tuning
            pred = results.get_prediction(start=-14)
            mae_check = mean_absolute_error(tune_y.tail(14), pred.predicted_mean)
            if mae_check < best_mae:
                best_mae = mae_check
                best_params = (param, param_seasonal)
        except:
            continue

print(f"Best Parameters Found: Order={best_params[0]}, Seasonal={best_params[1]}")

# -----------------------------
# 4. Walk-Forward Cross-Validation
# -----------------------------
# -----------------------------
# 4. Walk-Forward Cross-Validation (Corrected)
# -----------------------------
print("\n--- Starting Walk-Forward CV ---")
tscv_splits = 3
fold_size = 30
cv_results = []

# Total length of data we are using for CV (excluding the final holdout)
# We calculate manually to ensure slices never hit 0 or overlap incorrectly
total_len = len(y)

for i in range(tscv_splits):
    # Determine the split point: we want to leave the last (tscv_splits - i) * fold_size for testing
    # Fold 1: Train on up to -90, test -90 to -60
    # Fold 2: Train on up to -60, test -60 to -30
    # Fold 3: Train on up to -30, test -30 to end (Holdout equivalent)
    test_start = total_len - (fold_size * (tscv_splits - i))
    test_end = test_start + fold_size

    cv_train_y = y.iloc[:test_start]
    cv_test_y = y.iloc[test_start:test_end]

    cv_train_exog = exog.iloc[:test_start]
    cv_test_exog = exog.iloc[test_start:test_end]

    model_cv = SARIMAX(
        cv_train_y,
        exog=cv_train_exog,
        order=best_params[0],
        seasonal_order=best_params[1],
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    res_cv = model_cv.fit(disp=False)

    # We now explicitly pass the number of steps to avoid the "end after start" error
    forecast_cv = res_cv.get_forecast(
        steps=len(cv_test_y), exog=cv_test_exog
    ).predicted_mean

    fold_mae = mean_absolute_error(cv_test_y, forecast_cv)
    cv_results.append(fold_mae)
    print(
        f"Fold {i + 1} MAE: {fold_mae:.4f} (Test Range: {cv_test_y.index[0].date()} to {cv_test_y.index[-1].date()})"
    )

print(f"Average CV MAE: {np.mean(cv_results):.4f}")

# -----------------------------
# 5. Final Model Training & Forecast
# -----------------------------
split_date = y.index[-30]
train_y = y.loc[: split_date - pd.Timedelta(days=1)]
test_y = y.loc[split_date:]
train_exog = exog.loc[: split_date - pd.Timedelta(days=1)]
test_exog = exog.loc[split_date:]

final_model = SARIMAX(
    train_y,
    exog=train_exog,
    order=best_params[0],
    seasonal_order=best_params[1],
    enforce_stationarity=False,
    enforce_invertibility=False,
)
final_res = final_model.fit(disp=False)

forecast_final = final_res.get_forecast(steps=30, exog=test_exog)
y_pred = forecast_final.predicted_mean
conf_int = forecast_final.conf_int()

# -----------------------------
# 6. Evaluation Metrics
# -----------------------------
final_mae = mean_absolute_error(test_y, y_pred)
final_rmse = np.sqrt(mean_squared_error(test_y, y_pred))
final_mape = mean_absolute_percentage_error(test_y, y_pred)

print("\n--- FINAL HOLDOUT METRICS ---")
print(f"MAE:  {final_mae:.4f} million riders")
print(f"RMSE: {final_rmse:.4f} million riders")
print(f"MAPE: {final_mape:.2%}")

# -----------------------------
# 7. Residual Diagnostics
# -----------------------------
residuals = final_res.resid

# Ljung-Box test
lb_test = acorr_ljungbox(residuals, lags=[7, 14, 21], return_df=True)
print("\nLjung-Box Test:")
print(lb_test)

# Plot residuals
plt.figure(figsize=(16, 5))
plt.subplot(1, 3, 1)
plt.plot(residuals)
plt.title("Residuals Over Time")

plt.subplot(1, 3, 2)
plt.hist(residuals, bins=20)
plt.title("Histogram of Residuals")

plt.subplot(1, 3, 3)
stats.probplot(residuals, dist="norm", plot=plt)
plt.title("Q-Q Plot")
plt.tight_layout()
plt.show()

# -----------------------------
# 8. Forecast Plot
# -----------------------------
os.makedirs("reports/plots", exist_ok=True)
plt.figure(figsize=(12, 6))
plt.plot(test_y.index, test_y, label="Actual", color="orange")
plt.plot(test_y.index, y_pred, label="Tuned SARIMAX", linestyle="--", color="green")
plt.fill_between(
    test_y.index,
    conf_int.iloc[:, 0],
    conf_int.iloc[:, 1],
    color="gray",
    alpha=0.3,
    label="Confidence Interval",
)
plt.title(f"Tuned SARIMAX Forecast (MAPE: {final_mape:.2%})")
plt.legend()
plt.savefig("reports/plots/tuned_sarimax_final.png")
plt.show()
