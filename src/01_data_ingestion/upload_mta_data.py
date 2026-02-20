import os
import io
import sys
import requests
import boto3
from datetime import date, timedelta, datetime
from urllib.parse import quote
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd

from src.utils.logger import get_logger

# ------------------------------------------------------------------
# 1. Setup & Environment Validation
# ------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(dotenv_path=BASE_DIR / ".env")

logger = get_logger(__name__)

# Retrieve environment variables
AWS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET = os.getenv("AWS_SECRET_KEY")
REGION = os.getenv("AWS_REGION")
BUCKET = os.getenv("AWS_BUCKET_NAME")
APP_TOKEN = os.getenv("NY_APP_TOKEN")

required_vars = {
    "AWS_ACCESS_KEY": AWS_KEY,
    "AWS_SECRET_KEY": AWS_SECRET,
    "AWS_REGION": REGION,
    "AWS_BUCKET_NAME": BUCKET,
    "NY_APP_TOKEN": APP_TOKEN,
}

missing = [k for k, v in required_vars.items() if not v]
if missing:
    logger.error(f"Missing environment variables: {', '.join(missing)}")
    sys.exit(1)

S3_CLIENT = boto3.client(
    "s3",
    aws_access_key_id=AWS_KEY,
    aws_secret_access_key=AWS_SECRET,
    region_name=REGION,
)

DATASET_ID = "5wq4-mkjj"
BASE_URL = f"https://data.ny.gov/resource/{DATASET_ID}.csv"
META_FILE_KEY = "bronze/mta/last_fetched.txt"

# ------------------------------------------------------------------
# 2. Helper Functions
# ------------------------------------------------------------------


def build_soql_query(start_date: date, end_date: date, limit: int, offset: int) -> str:
    """Builds SoQL query for daily ridership with borough & station_complex, including pagination."""
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
    """Fetch all data for the given range, handling pagination, returns DataFrame."""

    all_rows = []
    limit = 500000
    offset = 0

    while True:
        soql = build_soql_query(start_date, end_date, limit, offset)
        url = f"{BASE_URL}?$query={soql}"
        headers = {"X-App-Token": APP_TOKEN}

        response = requests.get(url, headers=headers, timeout=300)
        response.raise_for_status()
        content = response.content.decode("utf-8")

        lines = content.splitlines()
        if offset > 0 and lines:
            lines = lines[1:]  # skip header for subsequent pages

        if not lines:
            break

        all_rows.extend(lines)
        logger.info(f"Fetched {len(lines)} rows (offset {offset})")

        if len(lines) < limit:
            break

        offset += limit

    if not all_rows:
        logger.warning("No data found for this range.")
        return pd.DataFrame()  # empty DF

    # Convert to DataFrame
    csv_content = "\n".join(all_rows)
    df = pd.read_csv(io.StringIO(csv_content))
    logger.info(f"Total rows fetched: {len(df)}")
    return df


def upload_to_s3(df: pd.DataFrame, min_date: date, max_date: date) -> None:
    """Upload DataFrame to S3 as one CSV file with min-max dates."""
    s3_key = f"bronze/mta/mta_daily_ridership_{min_date}_{max_date}.csv"
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    S3_CLIENT.put_object(Bucket=BUCKET, Key=s3_key, Body=csv_buffer.getvalue())
    logger.info(f"Uploaded: s3://{BUCKET}/{s3_key}")


def get_last_fetched_date() -> date:
    """Return the last fetched date from S3 watermark."""
    try:
        obj = S3_CLIENT.get_object(Bucket=BUCKET, Key=META_FILE_KEY)
        return datetime.strptime(obj["Body"].read().decode().strip(), "%Y-%m-%d").date()
    except S3_CLIENT.exceptions.NoSuchKey:
        logger.info("No watermark found. This is the first run.")
        return None


def update_last_fetched_date(max_date: date) -> None:
    """Update watermark file in S3."""
    S3_CLIENT.put_object(Bucket=BUCKET, Key=META_FILE_KEY, Body=str(max_date).encode())


# ------------------------------------------------------------------
# 3. Main Ingestion Function
# ------------------------------------------------------------------


def run():
    # Determine start date
    last_date = get_last_fetched_date()
    start_date = last_date + timedelta(days=1) if last_date else date(2025, 1, 1)

    # Use lag to ensure data is published (MTA updates Wednesdays)
    LAG_DAYS = 6
    end_date = date.today() - timedelta(days=LAG_DAYS)

    if start_date > end_date:
        logger.info("All data is up to date. Nothing to fetch.")
        return

    # Fetch all data
    df = fetch_mta_data(start_date, end_date)
    if df.empty:
        logger.warning("No new data available.")
        return

    # Use actual min and max dates from fetched data
    min_date = pd.to_datetime(df["transit_date"].min()).strftime("%Y-%m-%d")
    max_date = pd.to_datetime(df["transit_date"].max()).strftime("%Y-%m-%d")

    logger.info(f"Fetched data from {min_date} to {max_date} (actual)")

    # Upload single CSV
    upload_to_s3(df, min_date, max_date)

    # Update watermark
    update_last_fetched_date(max_date)

    logger.info(f"Ingestion complete. File uploaded for {min_date} to {max_date}")


# ------------------------------------------------------------------
# 4. Entrypoint
# ------------------------------------------------------------------

if __name__ == "__main__":
    run()
