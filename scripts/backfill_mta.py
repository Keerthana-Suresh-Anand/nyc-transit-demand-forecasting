"""
One-time backfill: fetch MTA subway ridership for 2022-2024 from the historical
dataset (wujg-7c2s) and write to S3 bronze. Does NOT touch the MTA watermark.
"""
import io
import sys
from datetime import date
from urllib.parse import quote

import pandas as pd
import requests

from src.utils.config import NY_APP_TOKEN, S3_MTA_PREFIX
from src.utils.logger import get_logger
from src.utils.s3_helpers import get_s3_client, list_s3_files, write_s3_csv

logger = get_logger(__name__)

HISTORICAL_DATASET_ID = "wujg-7c2s"
HISTORICAL_BASE_URL = f"https://data.ny.gov/resource/{HISTORICAL_DATASET_ID}.csv"

YEARS = [
    (date(2022, 1, 1), date(2022, 12, 31)),
    (date(2023, 1, 1), date(2023, 12, 31)),
    (date(2024, 1, 1), date(2024, 12, 31)),
]


def build_soql_query(start_date: date, end_date: date, limit: int, offset: int) -> str:
    query = (
        f"SELECT date_trunc_ymd(transit_timestamp) AS transit_date, "
        f"station_complex, borough, "
        f"sum(ridership) AS daily_ridership "
        f"WHERE transit_mode = 'subway' "
        f"AND transit_timestamp >= '{start_date}T00:00:00' "
        f"AND transit_timestamp <= '{end_date}T23:59:59' "
        f"GROUP BY transit_date, station_complex, borough "
        f"ORDER BY transit_date ASC "
        f"LIMIT {limit} OFFSET {offset}"
    )
    return quote(query)


def fetch_year(start_date: date, end_date: date) -> pd.DataFrame:
    all_rows = []
    limit = 500_000
    offset = 0

    while True:
        soql = build_soql_query(start_date, end_date, limit, offset)
        url = f"{HISTORICAL_BASE_URL}?$query={soql}"
        response = requests.get(url, headers={"X-App-Token": NY_APP_TOKEN}, timeout=300)
        response.raise_for_status()

        lines = response.content.decode("utf-8").splitlines()
        if offset > 0 and lines:
            lines = lines[1:]
        if not lines:
            break

        all_rows.extend(lines)
        logger.info(f"  Fetched {len(lines)} rows (offset {offset})")

        if len(lines) < limit:
            break
        offset += limit

    if not all_rows:
        return pd.DataFrame()

    return pd.read_csv(io.StringIO("\n".join(all_rows)))


def run() -> None:
    if not NY_APP_TOKEN:
        logger.error("NY_APP_TOKEN not set")
        sys.exit(1)

    s3 = get_s3_client()
    existing_keys = list_s3_files(s3, S3_MTA_PREFIX, extension=".csv")

    for start_date, end_date in YEARS:
        year = start_date.year
        s3_key = f"{S3_MTA_PREFIX}mta_daily_ridership_{start_date}_{end_date}.csv"

        if s3_key in existing_keys:
            logger.info(f"{year}: already exists in S3, skipping")
            continue

        logger.info(f"Fetching MTA data for {year} ({start_date} to {end_date})")
        df = fetch_year(start_date, end_date)

        if df.empty:
            logger.warning(f"{year}: no data returned")
            continue

        write_s3_csv(s3, df, s3_key)
        logger.info(f"{year}: {len(df):,} rows written to {s3_key}")

    logger.info("MTA backfill complete")


if __name__ == "__main__":
    run()
