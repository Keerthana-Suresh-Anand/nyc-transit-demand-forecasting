"""
One-time backfill: fetch historical weather for 2022-2024 from Visual Crossing
and write to S3 bronze. Does NOT touch the weather watermark.
"""
import io
import sys
import time
from datetime import date

import pandas as pd
import requests

from src.utils.config import (
    S3_WEATHER_HIST_PREFIX,
    WEATHER_API_KEY,
    WEATHER_BASE_URL,
    WEATHER_LOCATION,
)
from src.utils.logger import get_logger
from src.utils.s3_helpers import get_s3_client, list_s3_files, write_s3_csv

logger = get_logger(__name__)

YEARS = [
    (date(2022, 1, 1), date(2022, 12, 31)),
    (date(2023, 1, 1), date(2023, 12, 31)),
    (date(2024, 1, 1), date(2024, 12, 31)),
]


def fetch_weather(start_date: date, end_date: date, retries: int = 3) -> pd.DataFrame:
    url = (
        f"{WEATHER_BASE_URL}{WEATHER_LOCATION}/{start_date}/{end_date}"
        f"?unitGroup=metric&include=days&key={WEATHER_API_KEY}&contentType=csv"
    )
    for attempt in range(retries):
        response = requests.get(url, timeout=300)
        if response.status_code == 429:
            wait = 30 * (attempt + 1)
            logger.warning(f"Rate limited (429) — waiting {wait}s before retry {attempt + 1}/{retries}")
            time.sleep(wait)
            continue
        response.raise_for_status()
        df = pd.read_csv(io.StringIO(response.text))
        df["datetime"] = pd.to_datetime(df["datetime"]).dt.date
        return df
    raise RuntimeError(f"Failed to fetch weather for {start_date} to {end_date} after {retries} retries")


def run() -> None:
    if not WEATHER_API_KEY:
        logger.error("WEATHER_API_KEY not set")
        sys.exit(1)

    s3 = get_s3_client()
    existing_keys = list_s3_files(s3, S3_WEATHER_HIST_PREFIX, extension=".csv")

    for start_date, end_date in YEARS:
        year = start_date.year
        s3_key = f"{S3_WEATHER_HIST_PREFIX}weather_{start_date}_{end_date}.csv"

        if s3_key in existing_keys:
            logger.info(f"{year}: already exists in S3, skipping")
            continue

        logger.info(f"Fetching weather for {year} ({start_date} to {end_date})")
        df = fetch_weather(start_date, end_date)

        if df.empty:
            logger.warning(f"{year}: no data returned")
            continue

        write_s3_csv(s3, df, s3_key)
        logger.info(f"{year}: {len(df)} rows written to {s3_key}")
        time.sleep(5)

    logger.info("Weather backfill complete")


if __name__ == "__main__":
    run()
