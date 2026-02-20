import os
import io
import sys
import boto3
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

from src.utils.logger import get_logger

# ------------------------------------------------------------------
# 1. Setup & Environment
# ------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(dotenv_path=BASE_DIR / ".env")

logger = get_logger(__name__)

AWS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET = os.getenv("AWS_SECRET_KEY")
REGION = os.getenv("AWS_REGION")
BUCKET = os.getenv("AWS_BUCKET_NAME")

if not all([AWS_KEY, AWS_SECRET, REGION, BUCKET]):
    logger.error("Missing AWS credentials in .env file.")
    sys.exit(1)

S3_CLIENT = boto3.client(
    "s3",
    aws_access_key_id=AWS_KEY,
    aws_secret_access_key=AWS_SECRET,
    region_name=REGION,
)

BRONZE_MTA_PREFIX = "bronze/mta/"
BRONZE_WEATHER_PREFIX = "bronze/weather/historical/"

SILVER_LOCAL_PATH = BASE_DIR / "data" / "silver" / "mta_weather_merged.parquet"

# ------------------------------------------------------------------
# 2. Helper Functions
# ------------------------------------------------------------------


def get_end_date_from_filename(filename: str) -> pd.Timestamp:
    """
    Extracts end date from filenames like:
    mta_daily_ridership_2025-01-01_2026-01-29.csv
    weather_2025-01-01_2026-01-29.csv
    """
    try:
        clean_name = filename.replace(".csv", "").split("/")[-1]
        end_date_str = clean_name.split("_")[-1]
        return pd.to_datetime(end_date_str)
    except Exception:
        return pd.to_datetime("1900-01-01")


def list_s3_csv_files(prefix: str):
    paginator = S3_CLIENT.get_paginator("list_objects_v2")
    files = []
    for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".csv"):
                files.append(obj["Key"])
    return files


def read_s3_csv(key: str) -> pd.DataFrame:
    logger.info(f"Downloading: {key}")
    obj = S3_CLIENT.get_object(Bucket=BUCKET, Key=key)
    return pd.read_csv(
        io.StringIO(obj["Body"].read().decode("utf-8")), low_memory=False
    )


# ------------------------------------------------------------------
# 3. Merge Logic
# ------------------------------------------------------------------


def merge_mta_weather_incremental():

    # ---------------------------
    # Step 1: Determine watermark
    # ---------------------------

    if SILVER_LOCAL_PATH.exists():
        df_existing_dates = pd.read_parquet(SILVER_LOCAL_PATH, columns=["transit_date"])
        last_silver_date = pd.to_datetime(df_existing_dates["transit_date"]).max()

        logger.info(f"Existing Silver found. Latest date: {last_silver_date.date()}")
        incremental = True
    else:
        last_silver_date = pd.to_datetime("1900-01-01")
        incremental = False
        logger.info("No Silver file found. Running full build.")

    # ---------------------------
    # Step 2: Smart MTA Filtering
    # ---------------------------

    all_mta_keys = list_s3_csv_files(BRONZE_MTA_PREFIX)

    mta_keys_to_pull = [
        k for k in all_mta_keys if get_end_date_from_filename(k) > last_silver_date
    ]

    if not mta_keys_to_pull:
        logger.info("No new MTA files to process.")
        return

    logger.info(f"Processing {len(mta_keys_to_pull)} MTA files.")

    df_new_mta = pd.concat(
        [read_s3_csv(k) for k in mta_keys_to_pull], ignore_index=True
    )

    df_new_mta["transit_date"] = pd.to_datetime(df_new_mta["transit_date"])

    # Row-level protection
    df_new_mta = df_new_mta[df_new_mta["transit_date"] > last_silver_date]

    if df_new_mta.empty:
        logger.info("All MTA records already exist in Silver.")
        return

    # ---------------------------
    # Step 3: Smart Weather Filtering
    # ---------------------------

    all_weather_keys = list_s3_csv_files(BRONZE_WEATHER_PREFIX)

    weather_keys_to_pull = [
        k for k in all_weather_keys if get_end_date_from_filename(k) > last_silver_date
    ]

    if not weather_keys_to_pull:
        logger.warning("No relevant weather files found.")
        return

    logger.info(f"Processing {len(weather_keys_to_pull)} weather files.")

    df_weather = pd.concat(
        [read_s3_csv(k) for k in weather_keys_to_pull], ignore_index=True
    )

    df_weather["datetime"] = pd.to_datetime(df_weather["datetime"])

    # Only keep weather for new MTA dates
    relevant_dates = df_new_mta["transit_date"].unique()

    df_weather = df_weather[df_weather["datetime"].isin(relevant_dates)]

    # ---------------------------
    # Step 4: Merge
    # ---------------------------

    logger.info(f"Merging {len(df_new_mta)} MTA rows with weather.")

    df_merged = pd.merge(
        df_new_mta, df_weather, left_on="transit_date", right_on="datetime", how="left"
    )

    if "datetime" in df_merged.columns:
        df_merged.drop(columns=["datetime"], inplace=True)

    # ---------------------------
    # Step 5: Combine + Clean
    # ---------------------------

    if incremental:
        df_old = pd.read_parquet(SILVER_LOCAL_PATH)
        df_final = pd.concat([df_old, df_merged], ignore_index=True)
    else:
        df_final = df_merged

    # Deduplicate safeguard
    df_final = df_final.drop_duplicates()

    # Ensure time ordering
    df_final = df_final.sort_values("transit_date").reset_index(drop=True)

    # ---------------------------
    # Step 6: Save Locally
    # ---------------------------

    SILVER_LOCAL_PATH.parent.mkdir(parents=True, exist_ok=True)

    df_final.to_parquet(SILVER_LOCAL_PATH, index=False, engine="pyarrow")

    logger.info("Silver updated successfully.")
    logger.info(f"Total rows: {len(df_final)}")
    logger.info(f"Latest date: {df_final['transit_date'].max().date()}")


if __name__ == "__main__":
    merge_mta_weather_incremental()
