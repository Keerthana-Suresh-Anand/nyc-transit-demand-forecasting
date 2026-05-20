"""Tests for weather ingestion: fetch_weather and run() orchestration."""
from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import boto3
import pandas as pd
import pytest
from moto import mock_aws

from src.ingestion.ingest_weather import fetch_weather, run
from src.utils.config import S3_MTA_WATERMARK, S3_WEATHER_FORECAST_PREFIX, S3_WEATHER_WATERMARK

BUCKET = "test-bucket"


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


def _make_weather_response(n_days: int = 5) -> MagicMock:
    header = "datetime,temp,precip,snow,humidity"
    rows = [f"2025-01-{i+1:02d},10.0,0.0,0.0,60" for i in range(n_days)]
    mock_resp = MagicMock()
    mock_resp.text = "\n".join([header] + rows)
    mock_resp.raise_for_status.return_value = None
    return mock_resp


class TestFetchWeather:
    def test_returns_dataframe_with_datetime_column(self):
        with patch("src.ingestion.ingest_weather.requests.get", return_value=_make_weather_response(3)):
            df = fetch_weather(date(2025, 1, 1), date(2025, 1, 3))
        assert "datetime" in df.columns
        assert len(df) == 3

    def test_datetime_column_converted_to_date(self):
        with patch("src.ingestion.ingest_weather.requests.get", return_value=_make_weather_response(2)):
            df = fetch_weather(date(2025, 1, 1), date(2025, 1, 2))
        assert isinstance(df["datetime"].iloc[0], date)

    def test_raises_on_http_error(self):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = Exception("HTTP 500")
        with patch("src.ingestion.ingest_weather.requests.get", return_value=mock_resp):
            with pytest.raises(Exception):
                fetch_weather(date(2025, 1, 1), date(2025, 1, 5))

    def test_exits_when_api_key_missing(self, monkeypatch):
        monkeypatch.setattr("src.ingestion.ingest_weather.WEATHER_API_KEY", "")
        with pytest.raises(SystemExit):
            fetch_weather(date(2025, 1, 1), date(2025, 1, 5))


class TestIngestWeatherRun:
    def test_exits_when_mta_watermark_missing(self, s3, monkeypatch):
        monkeypatch.setattr("src.ingestion.ingest_weather.get_s3_client", lambda: s3)
        with pytest.raises(SystemExit):
            run()

    def test_always_writes_forecast_csv(self, s3, monkeypatch):
        monkeypatch.setattr("src.ingestion.ingest_weather.get_s3_client", lambda: s3)
        # Set MTA watermark so run() proceeds
        mta_date = date.today() - timedelta(days=2)
        s3.put_object(Bucket=BUCKET, Key=S3_MTA_WATERMARK, Body=str(mta_date).encode())
        # Set weather watermark = mta date so historical is skipped
        s3.put_object(Bucket=BUCKET, Key=S3_WEATHER_WATERMARK, Body=str(mta_date).encode())

        monkeypatch.setattr(
            "src.ingestion.ingest_weather.fetch_weather",
            lambda s, e: pd.DataFrame({"datetime": [s], "temp": [10.0], "precip": [0.0], "snow": [0.0]}),
        )

        run()

        from src.utils.s3_helpers import list_s3_files
        forecast_files = list_s3_files(s3, S3_WEATHER_FORECAST_PREFIX, ".csv")
        assert len(forecast_files) >= 1

    def test_fetches_historical_when_weather_behind_mta(self, s3, monkeypatch):
        monkeypatch.setattr("src.ingestion.ingest_weather.get_s3_client", lambda: s3)
        mta_date = date(2025, 3, 10)
        weather_date = date(2025, 3, 1)
        s3.put_object(Bucket=BUCKET, Key=S3_MTA_WATERMARK, Body=str(mta_date).encode())
        s3.put_object(Bucket=BUCKET, Key=S3_WEATHER_WATERMARK, Body=str(weather_date).encode())

        fetch_calls = []

        def capture_fetch(start, end):
            fetch_calls.append((start, end))
            return pd.DataFrame({
                "datetime": [start],
                "temp": [10.0], "precip": [0.0], "snow": [0.0],
            })

        monkeypatch.setattr("src.ingestion.ingest_weather.fetch_weather", capture_fetch)

        run()

        # At least two calls: one historical + one forecast
        assert len(fetch_calls) >= 2
