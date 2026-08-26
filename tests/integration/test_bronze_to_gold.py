"""Integration: bronze CSVs on (mocked) S3 → silver merge → both gold layers.

Exercises the real ``merge_silver`` → ``preprocess_sarima`` → ``preprocess_ml``
chain end-to-end: S3 listing/reading, the station→city aggregation, holiday and
lag feature construction, and the parquet handoffs between stages.
"""
import boto3
import pandas as pd
import pytest
from moto import mock_aws

from src.ingestion import merge_silver
from src.transformation import preprocess_ml, preprocess_sarima

BUCKET = "test-bucket"
N_DAYS = 40  # enough for the 14-day lag/rolling features to survive dropna


@pytest.fixture()
def s3():
    with mock_aws():
        client = boto3.client(
            "s3",
            region_name="us-east-1",
            aws_access_key_id="test-key",
            aws_secret_access_key="test-secret",
        )
        client.create_bucket(Bucket=BUCKET)
        yield client


@pytest.fixture()
def local_paths(tmp_path, monkeypatch):
    """Point every stage's local parquet paths into an isolated tmp dir."""
    silver = tmp_path / "silver" / "mta_weather_merged.parquet"
    gold_sarima = tmp_path / "gold" / "mta_sarima.parquet"
    gold_ml = tmp_path / "gold" / "mta_ml.parquet"
    monkeypatch.setattr(merge_silver, "SILVER_LOCAL_PATH", silver)
    monkeypatch.setattr(preprocess_sarima, "SILVER_LOCAL_PATH", silver)
    monkeypatch.setattr(preprocess_sarima, "GOLD_SARIMA_LOCAL_PATH", gold_sarima)
    monkeypatch.setattr(preprocess_ml, "GOLD_SARIMA_LOCAL_PATH", gold_sarima)
    monkeypatch.setattr(preprocess_ml, "GOLD_ML_LOCAL_PATH", gold_ml)
    return {"silver": silver, "gold_sarima": gold_sarima, "gold_ml": gold_ml}


def _seed_bronze(s3):
    dates = pd.date_range("2025-06-01", periods=N_DAYS, freq="D")
    mta = pd.DataFrame({
        "transit_date": list(dates) * 2,
        "station_complex": ["Grand Central"] * N_DAYS + ["Times Sq"] * N_DAYS,
        "borough": ["Manhattan"] * (2 * N_DAYS),
        "daily_ridership": [1_500_000 + i * 1_000 for i in range(N_DAYS)] * 2,
    })
    weather = pd.DataFrame({
        "datetime": dates.date,
        "temp": [20.0 + (i % 10) for i in range(N_DAYS)],
        "precip": [0.0] * N_DAYS,
        "snow": [0.0] * (N_DAYS - 1) + [5.0],
    })
    end = str(dates.max().date())
    s3.put_object(
        Bucket=BUCKET,
        Key=f"bronze/mta/mta_daily_ridership_2025-06-01_{end}.csv",
        Body=mta.to_csv(index=False),
    )
    s3.put_object(
        Bucket=BUCKET,
        Key=f"bronze/weather/historical/weather_2025-06-01_{end}.csv",
        Body=weather.to_csv(index=False),
    )


def test_bronze_to_gold(s3, local_paths, monkeypatch):
    monkeypatch.setattr(merge_silver, "get_s3_client", lambda: s3)
    _seed_bronze(s3)

    merge_silver.run()
    gold_sarima = preprocess_sarima.run()
    gold_ml = preprocess_ml.run()

    # Silver: one row per station per day, weather joined on
    silver = pd.read_parquet(local_paths["silver"])
    assert len(silver) == 2 * N_DAYS
    assert {"transit_date", "daily_ridership", "temp", "precip", "snow"}.issubset(silver.columns)

    # Gold SARIMA: daily city-wide aggregate with engineered exog
    assert len(gold_sarima) == N_DAYS
    first_day = gold_sarima.iloc[0]
    assert first_day["daily_ridership"] == 2 * 1_500_000  # both stations summed
    assert gold_sarima["is_holiday"].isin([0, 1]).all()
    assert gold_sarima["snow_lag1"].iloc[-1] == 0.0  # snow on last day lags out of frame

    # Gold ML: lag/rolling features present and consistent with the aggregate
    assert {"ridership_lag1", "ridership_lag7", "ridership_14d_avg", "day_of_week"}.issubset(gold_ml.columns)
    assert not gold_ml.isna().any().any()  # dropna removed the lag warm-up rows
    last = gold_ml.iloc[-1]
    prev_actual_m = gold_sarima["daily_ridership"].iloc[-2] / 1_000_000
    assert last["ridership_lag1"] == pytest.approx(prev_actual_m)

    # Persisted gold files match what the stage functions returned
    assert pd.read_parquet(local_paths["gold_sarima"]).equals(gold_sarima)
    assert pd.read_parquet(local_paths["gold_ml"]).equals(gold_ml)
