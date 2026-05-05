import io
import sys
from datetime import date, timedelta

import pandas as pd
import requests

from src.utils.config import (
    S3_MTA_WATERMARK,
    S3_WEATHER_FORECAST_PREFIX,
    S3_WEATHER_HIST_PREFIX,
    S3_WEATHER_WATERMARK,
    WEATHER_API_KEY,
    WEATHER_BASE_URL,
    WEATHER_FORECAST_DAYS,
    WEATHER_LOCATION,
)
from src.utils.logger import get_logger
from src.utils.s3_helpers import get_s3_client, read_watermark, write_s3_csv, write_watermark

logger = get_logger(__name__)


def fetch_weather(start_date: date, end_date: date) -> pd.DataFrame:
    if not WEATHER_API_KEY:
        logger.error("WEATHER_API_KEY not set")
        sys.exit(1)

    url = (
        f"{WEATHER_BASE_URL}{WEATHER_LOCATION}/{start_date}/{end_date}"
        f"?unitGroup=metric&include=days&key={WEATHER_API_KEY}&contentType=csv"
    )
    logger.info(f"Fetching weather: {start_date} to {end_date}")
    response = requests.get(url, timeout=300)
    response.raise_for_status()

    df = pd.read_csv(io.StringIO(response.text))
    df["datetime"] = pd.to_datetime(df["datetime"]).dt.date
    logger.info(f"Fetched {len(df)} weather rows")
    return df


def run() -> None:
    logger.info("Starting weather ingestion")
    s3 = get_s3_client()

    mta_last_date = read_watermark(s3, S3_MTA_WATERMARK)
    if not mta_last_date:
        logger.error("MTA watermark not found. Run MTA ingestion first.")
        sys.exit(1)

    weather_last_date = read_watermark(s3, S3_WEATHER_WATERMARK)
    start_date = (weather_last_date + timedelta(days=1)) if weather_last_date else date(2025, 1, 1)
    end_date = mta_last_date

    if start_date <= end_date:
        df_hist = fetch_weather(start_date, end_date)
        min_date = df_hist["datetime"].min()
        max_date = df_hist["datetime"].max()
        hist_key = f"{S3_WEATHER_HIST_PREFIX}weather_{min_date}_{max_date}.csv"
        write_s3_csv(s3, df_hist, hist_key)
        write_watermark(s3, S3_WEATHER_WATERMARK, max_date)
        logger.info(f"Historical weather stored: {min_date} to {max_date}")
    else:
        logger.info("No new historical weather needed")

    forecast_start = date.today()
    forecast_end = forecast_start + timedelta(days=WEATHER_FORECAST_DAYS)
    df_forecast = fetch_weather(forecast_start, forecast_end)
    forecast_key = f"{S3_WEATHER_FORECAST_PREFIX}weather_forecast_run_{date.today()}.csv"
    write_s3_csv(s3, df_forecast, forecast_key)

    logger.info(f"Forecast stored: {forecast_start} to {forecast_end}")
    logger.info("Weather ingestion complete")


if __name__ == "__main__":
    run()
