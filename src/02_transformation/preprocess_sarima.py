import pandas as pd
import holidays
import numpy as np
import os

# 1. Load Data
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Build the path relative to the Project Root
data_path = os.path.join(BASE_DIR, "data", "silver", "mta_weather_merged.parquet")
df = pd.read_parquet(data_path)

# 2. Daily Aggregation (City-wide level)
# We aggregate to focus on the overall system heartbeat rather than individual stations
df_daily = (
    df.groupby("transit_date")
    .agg({"daily_ridership": "sum", "temp": "mean", "precip": "mean", "snow": "mean"})
    .sort_index()
)

# 3. Ensure Strict Daily Frequency
# SARIMA requires a continuous timeline; this fills any missing days with NaNs
df_daily = df_daily.asfreq("D")

# 4. Handle Missing Values (Imputation)
# Linear interpolation for ridership/temp to maintain the trend
df_daily["daily_ridership"] = df_daily["daily_ridership"].interpolate(method="linear")
df_daily["temp"] = df_daily["temp"].interpolate(method="linear")
# Fill precip/snow with 0 (assume no data means no precipitation)
df_daily[["precip", "snow"]] = df_daily[["precip", "snow"]].fillna(0)

# 5. Holiday Feature Engineering
# Using the holidays library to flag the "Special Days" identified in EDA
us_holidays = holidays.US(years=[2024, 2025, 2026])
df_daily["is_holiday"] = df_daily.index.map(lambda x: 1 if x in us_holidays else 0)

# 6. Feature Lagging
# Snow often impacts ridership the day after the storm (cleanup/slush)
df_daily["snow_lag1"] = df_daily["snow"].shift(1).fillna(0)

# 7. Save "Gold" Data
BASE_DIR = os.getcwd()
gold_dir = os.path.join(BASE_DIR, "data", "gold")

# Create the folder if it doesn't exist
os.makedirs(gold_dir, exist_ok=True)

gold_path = os.path.join(gold_dir, "mta_sarima.parquet")
df_daily.to_parquet(gold_path)

print(f"Success! Data saved to {gold_path}")
print(f"Total Days: {len(df_daily)}")
print(f"Holidays Flagged: {df_daily['is_holiday'].sum()}")
