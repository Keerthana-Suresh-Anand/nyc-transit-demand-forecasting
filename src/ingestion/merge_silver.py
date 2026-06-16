import pandas as pd

from src.utils.config import (
    S3_MTA_PREFIX,
    S3_WEATHER_HIST_PREFIX,
    SILVER_LOCAL_PATH,
)
from src.utils.logger import get_logger
from src.utils.s3_helpers import (
    get_end_date_from_filename,
    get_s3_client,
    list_s3_files,
    read_s3_csv,
)

logger = get_logger(__name__)

_REQUIRED_MTA_COLS = {"transit_date", "daily_ridership"}
_REQUIRED_WEATHER_COLS = {"datetime", "temp", "precip", "snow"}


def run() -> None:
    logger.info("Starting silver merge")
    s3 = get_s3_client()

    if SILVER_LOCAL_PATH.exists():
        existing_dates = pd.read_parquet(SILVER_LOCAL_PATH, columns=["transit_date"])
        last_silver_date = pd.to_datetime(existing_dates["transit_date"]).max()
        incremental = True
        logger.info(f"Existing silver found. Latest date: {last_silver_date.date()}")
    else:
        last_silver_date = pd.to_datetime("1900-01-01")
        incremental = False
        logger.info("No silver file found. Running full build.")

    all_mta_keys = list_s3_files(s3, S3_MTA_PREFIX)
    mta_keys = [k for k in all_mta_keys if get_end_date_from_filename(k) > last_silver_date]

    if not mta_keys:
        logger.info("No new MTA files to process.")
        return

    logger.info(f"Processing {len(mta_keys)} MTA file(s)")
    df_new_mta = pd.concat([read_s3_csv(s3, k) for k in mta_keys], ignore_index=True)

    missing_mta = _REQUIRED_MTA_COLS - set(df_new_mta.columns)
    if missing_mta:
        logger.error(f"MTA data missing required columns: {missing_mta} — aborting silver merge")
        return

    df_new_mta["transit_date"] = pd.to_datetime(df_new_mta["transit_date"])
    df_new_mta = df_new_mta[df_new_mta["transit_date"] > last_silver_date]

    if df_new_mta.empty:
        logger.info("All MTA records already in silver.")
        return

    all_weather_keys = list_s3_files(s3, S3_WEATHER_HIST_PREFIX)
    weather_keys = [k for k in all_weather_keys if get_end_date_from_filename(k) > last_silver_date]

    if not weather_keys:
        logger.warning("No relevant weather files found.")
        return

    logger.info(f"Processing {len(weather_keys)} weather file(s)")
    df_weather = pd.concat([read_s3_csv(s3, k) for k in weather_keys], ignore_index=True)

    missing_weather = _REQUIRED_WEATHER_COLS - set(df_weather.columns)
    if missing_weather:
        logger.error(f"Weather data missing required columns: {missing_weather} — aborting silver merge")
        return

    df_weather["datetime"] = pd.to_datetime(df_weather["datetime"])

    relevant_dates = df_new_mta["transit_date"].unique()
    df_weather = df_weather[df_weather["datetime"].isin(relevant_dates)]

    df_merged = pd.merge(
        df_new_mta, df_weather, left_on="transit_date", right_on="datetime", how="left"
    )
    df_merged.drop(columns=["datetime"], errors="ignore", inplace=True)

    if incremental:
        df_old = pd.read_parquet(SILVER_LOCAL_PATH)
        df_final = pd.concat([df_old, df_merged], ignore_index=True)
    else:
        df_final = df_merged

    df_final = df_final.drop_duplicates().sort_values("transit_date").reset_index(drop=True)

    SILVER_LOCAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_parquet(SILVER_LOCAL_PATH, index=False, engine="pyarrow")

    logger.info(f"Silver updated: {len(df_final):,} rows, latest: {df_final['transit_date'].max().date()}")


if __name__ == "__main__":
    run()
