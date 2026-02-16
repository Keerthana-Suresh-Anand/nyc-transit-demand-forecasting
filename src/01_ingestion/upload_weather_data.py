import os
import sys
import io
import requests
import boto3
import pandas as pd
from datetime import date, timedelta, datetime
from pathlib import Path
from dotenv import load_dotenv

from src.utils.logger import get_logger

# ------------------------------------------------------------------
# 1. Setup
# ------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(dotenv_path=BASE_DIR / ".env")

logger = get_logger(__name__)

WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
AWS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET = os.getenv("AWS_SECRET_KEY")
REGION = os.getenv("AWS_REGION")
BUCKET = os.getenv("AWS_BUCKET_NAME")

required = {
    "WEATHER_API_KEY": WEATHER_API_KEY,
    "AWS_ACCESS_KEY": AWS_KEY,
    "AWS_SECRET_KEY": AWS_SECRET,
    "AWS_REGION": REGION,
    "AWS_BUCKET_NAME": BUCKET,
}

missing = [k for k, v in required.items() if not v]
if missing:
    logger.error(f"Missing environment variables: {', '.join(missing)}")
    sys.exit(1)

S3_CLIENT = boto3.client(
    "s3",
    aws_access_key_id=AWS_KEY,
    aws_secret_access_key=AWS_SECRET,
    region_name=REGION,
)

LOCATION = "40.7812,-73.9665"  # NYC
BASE_URL = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/"
WEATHER_WATERMARK = "bronze/weather/last_fetched.txt"
MTA_WATERMARK = "bronze/mta/last_fetched.txt"

# ------------------------------------------------------------------
# 2. Helpers
# ------------------------------------------------------------------


def get_s3_date(key):
    try:
        obj = S3_CLIENT.get_object(Bucket=BUCKET, Key=key)
        return datetime.strptime(obj["Body"].read().decode().strip(), "%Y-%m-%d").date()
    except S3_CLIENT.exceptions.NoSuchKey:
        return None


def update_s3_date(key, value):
    S3_CLIENT.put_object(Bucket=BUCKET, Key=key, Body=str(value).encode())


def fetch_weather(start_date: date, end_date: date) -> pd.DataFrame:
    logger.info(f"Fetching weather from {start_date} to {end_date}")

    url = (
        f"{BASE_URL}{LOCATION}/{start_date}/{end_date}"
        f"?unitGroup=metric&include=days"
        f"&key={WEATHER_API_KEY}&contentType=csv"
    )

    response = requests.get(url, timeout=300)
    response.raise_for_status()

    df = pd.read_csv(io.StringIO(response.text))
    df["datetime"] = pd.to_datetime(df["datetime"]).dt.date

    logger.info(f"Fetched {len(df)} weather rows")
    return df


def upload_df(df: pd.DataFrame, key: str):
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    S3_CLIENT.put_object(Bucket=BUCKET, Key=key, Body=buffer.getvalue())
    logger.info(f"Uploaded: s3://{BUCKET}/{key}")


# ------------------------------------------------------------------
# 3. Main Logic
# ------------------------------------------------------------------


def run():

    logger.info("Starting weather ingestion")

    mta_last_date = get_s3_date(MTA_WATERMARK)
    if not mta_last_date:
        logger.error("MTA watermark not found. Run MTA ingestion first.")
        return

    weather_last_date = get_s3_date(WEATHER_WATERMARK)

    # --------------------------
    # HISTORICAL WEATHER
    # --------------------------

    if not weather_last_date:
        start_date = date(2025, 1, 1)
        logger.info("First weather run detected")
    else:
        start_date = weather_last_date + timedelta(days=1)

    end_date = mta_last_date

    if start_date <= end_date:
        df_hist = fetch_weather(start_date, end_date)

        min_date = df_hist["datetime"].min()
        max_date = df_hist["datetime"].max()

        hist_key = f"bronze/weather/historical/weather_{min_date}_{max_date}.csv"

        upload_df(df_hist, hist_key)
        update_s3_date(WEATHER_WATERMARK, max_date)

        logger.info(f"Historical weather stored: {min_date} to {max_date}")
    else:
        logger.info("No new historical weather needed")

    # --------------------------
    # FUTURE 7-DAY FORECAST
    # --------------------------

    forecast_start = date.today()
    forecast_end = forecast_start + timedelta(days=7)

    df_forecast = fetch_weather(forecast_start, forecast_end)

    run_date = date.today()

    forecast_key = f"bronze/weather/forecast/weather_forecast_run_{run_date}.csv"

    upload_df(df_forecast, forecast_key)

    logger.info(f"Forecast weather stored for {forecast_start} to {forecast_end}")

    logger.info("Weather ingestion complete")


# ------------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------------

if __name__ == "__main__":
    run()
