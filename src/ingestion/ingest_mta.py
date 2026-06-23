import io
from datetime import date, timedelta
from urllib.parse import quote

import pandas as pd
import requests

from src.utils.config import (
    MTA_BASE_URL,
    MTA_HISTORICAL_DATASET_ID,
    MTA_HISTORICAL_END_DATE,
    MTA_LAG_DAYS,
    MTA_START_DATE,
    NY_APP_TOKEN,
    S3_MTA_PREFIX,
    S3_MTA_WATERMARK,
)
from src.utils.logger import get_logger
from src.utils.s3_helpers import (
    MissingCredentialsError,
    get_s3_client,
    read_watermark,
    write_s3_csv,
    write_watermark,
)

logger = get_logger(__name__)

_HISTORICAL_END = date.fromisoformat(MTA_HISTORICAL_END_DATE)
_HISTORICAL_BASE_URL = f"https://data.ny.gov/resource/{MTA_HISTORICAL_DATASET_ID}.csv"


def _base_url_for(fetch_date: date) -> str:
    return _HISTORICAL_BASE_URL if fetch_date <= _HISTORICAL_END else MTA_BASE_URL


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
    """Fetch MTA data, splitting at the historical/current dataset boundary if needed."""
    if not NY_APP_TOKEN:
        raise MissingCredentialsError("NY_APP_TOKEN not set")

    # Split fetch window at the dataset boundary if it spans both
    segments = []
    if start_date <= _HISTORICAL_END and end_date > _HISTORICAL_END:
        segments = [
            (start_date, _HISTORICAL_END, _HISTORICAL_BASE_URL),
            (date(2025, 1, 1), end_date, MTA_BASE_URL),
        ]
    elif start_date <= _HISTORICAL_END:
        segments = [(start_date, end_date, _HISTORICAL_BASE_URL)]
    else:
        segments = [(start_date, end_date, MTA_BASE_URL)]

    all_dfs = []
    for seg_start, seg_end, base_url in segments:
        df = _fetch_segment(seg_start, seg_end, base_url)
        if not df.empty:
            all_dfs.append(df)

    if not all_dfs:
        return pd.DataFrame()
    return pd.concat(all_dfs, ignore_index=True)


def _fetch_segment(start_date: date, end_date: date, base_url: str) -> pd.DataFrame:
    all_rows = []
    limit = 500_000
    offset = 0

    while True:
        soql = build_soql_query(start_date, end_date, limit, offset)
        url = f"{base_url}?$query={soql}"
        response = requests.get(url, headers={"X-App-Token": NY_APP_TOKEN}, timeout=300)
        response.raise_for_status()

        lines = response.content.decode("utf-8").splitlines()
        if offset > 0 and lines:
            lines = lines[1:]
        if not lines:
            break

        all_rows.extend(lines)
        logger.info(f"Fetched {len(lines)} rows (offset {offset}) from {base_url.split('/')[4]}")

        if len(lines) < limit:
            break
        offset += limit

    if not all_rows:
        logger.warning(f"No data found for {start_date} to {end_date}")
        return pd.DataFrame()

    df = pd.read_csv(io.StringIO("\n".join(all_rows)))
    logger.info(f"Segment total: {len(df)} rows ({start_date} to {end_date})")
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
