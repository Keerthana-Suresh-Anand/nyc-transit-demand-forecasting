"""
Shared fixtures for the test suite.

Env vars are set at module level (before any imports) so that src/utils/config.py
reads them when it is first imported during test collection.
"""
import os

# Set before any src.* import — config.py reads these at load time
os.environ.setdefault("AWS_ACCESS_KEY", "test-key")
os.environ.setdefault("AWS_SECRET_KEY", "test-secret")
os.environ.setdefault("AWS_BUCKET_NAME", "test-bucket")
os.environ.setdefault("AWS_REGION", "us-east-1")
os.environ.setdefault("NY_APP_TOKEN", "test-token")
os.environ.setdefault("WEATHER_API_KEY", "test-weather-key")
os.environ.setdefault("TICKETMASTER_API_KEY", "test-tm-key")

import pandas as pd
import pytest


@pytest.fixture()
def sample_silver_df() -> pd.DataFrame:
    """Minimal silver schema: one row per station per day."""
    dates = pd.date_range("2025-01-01", periods=30, freq="D")
    return pd.DataFrame({
        "transit_date": dates.tolist() * 2,
        "station_complex": ["Grand Central-42 St"] * 30 + ["Times Sq-42 St"] * 30,
        "borough": ["Manhattan"] * 60,
        "daily_ridership": [50_000 + i * 100 for i in range(60)],
        "temp": [40.0 + i * 0.3 for i in range(60)],
        "precip": [0.0] * 55 + [0.5, 1.2, 0.0, 0.0, 0.0],
        "snow": [0.0] * 60,
    })


@pytest.fixture()
def sample_gold_sarima_df() -> pd.DataFrame:
    """Daily city-wide gold SARIMA schema."""
    dates = pd.date_range("2025-01-01", periods=100, freq="D")
    df = pd.DataFrame({
        "daily_ridership": [3_000_000 + i * 5_000 for i in range(100)],
        "temp": [40.0 + i * 0.2 for i in range(100)],
        "precip": [0.0] * 95 + [0.5, 1.2, 0.0, 0.0, 0.0],
        "snow": [0.0] * 100,
        "is_holiday": [0] * 100,
        "snow_lag1": [0.0] * 100,
    }, index=dates)
    df.index.name = "transit_date"
    return df
