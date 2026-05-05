import io
import sys
from datetime import date, timedelta
from urllib.parse import quote

import pandas as pd
import requests

from src.utils.config import (
    MTA_BASE_URL, MTA_LAG_DAYS, MTA_START_DATE,
    NY_APP_TOKEN, S3_MTA_PREFIX, S3_MTA_WATERMARK,
)
from src.utils.logger import get_logger
from src.utils.s3_helpers import (
    get_s3_client, get_end_date_from_filename, read_watermark,
    write_watermark, write_s3_csv,
)

logger = get_logger(__name__)


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


def fetch_mta_data(start_date: date, end_date: date) -> pd.DataFrame:
    if not NY_APP_TOKEN:
        logger.error("NY_APP_TOKEN not set")
        sys.exit(1)

    all_rows = []
    limit = 500_000
    offset = 0

    while True:
        soql = build_soql_query(start_date, end_date, limit, offset)
        url = f"{MTA_BASE_URL}?$query={soql}"
        response = requests.get(url, headers={"X-App-Token": NY_APP_TOKEN}, timeout=300)
        response.raise_for_status()

        lines = response.content.decode("utf-8").splitlines()
        if offset > 0 and lines:
            lines = lines[1:]  # skip header on subsequent pages
        if not lines:
            break

        all_rows.extend(lines)
        logger.info(f"Fetched {len(lines)} rows (offset {offset})")

        if len(lines) < limit:
            break
        offset += limit

    if not all_rows:
        logger.warning("No data found for this range.")
        return pd.DataFrame()

    df = pd.read_csv(io.StringIO("\n".join(all_rows)))
    logger.info(f"Total rows fetched: {len(df)}")
    return df


def run() -> None:
    logger.info("Starting MTA ingestion")
    s3 = get_s3_client()

    last_date = read_watermark(s3, S3_MTA_WATERMARK)
    start_date = (last_date + timedelta(days=1)) if last_date else date.fromisoformat(MTA_START_DATE)
    end_date = date.today() - timedelta(days=MTA_LAG_DAYS)

    if start_date > end_date:
        logger.info("MTA data is up to date. Nothing to fetch.")
        return

    logger.info(f"Fetching MTA data: {start_date} to {end_date}")
    df = fetch_mta_data(start_date, end_date)

    if df.empty:
        logger.warning("No new MTA data available.")
        return

    min_date = pd.to_datetime(df["transit_date"].min()).strftime("%Y-%m-%d")
    max_date = pd.to_datetime(df["transit_date"].max()).strftime("%Y-%m-%d")

    s3_key = f"{S3_MTA_PREFIX}mta_daily_ridership_{min_date}_{max_date}.csv"
    write_s3_csv(s3, df, s3_key)
    write_watermark(s3, S3_MTA_WATERMARK, max_date)

    logger.info(f"MTA ingestion complete: {len(df):,} rows, {min_date} to {max_date}")


if __name__ == "__main__":
    run()
