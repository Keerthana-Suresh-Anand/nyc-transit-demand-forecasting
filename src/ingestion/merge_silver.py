import pandas as pd

from src.utils.config import (
    MTA_LOOKBACK_DAYS,
    S3_MTA_PREFIX,
    S3_WEATHER_HIST_PREFIX,
    SILVER_LOCAL_PATH,
)
from src.utils.logger import get_logger
from src.utils.s3_helpers import (
    get_end_date_from_filename,
    get_s3_client,
    list_s3_files,
    list_s3_files_with_mtime,
    read_s3_csv,
)

logger = get_logger(__name__)

_REQUIRED_MTA_COLS = {"transit_date", "daily_ridership"}
_REQUIRED_WEATHER_COLS = {"datetime", "temp", "precip", "snow"}


def join_weather(df_mta: pd.DataFrame, df_weather: pd.DataFrame) -> pd.DataFrame:
    """Attach daily weather to station-level ridership rows. Left join — a missing
    weather day must not drop the ridership it belongs to."""
    merged = pd.merge(
        df_mta, df_weather, left_on="transit_date", right_on="datetime", how="left"
    )
    return merged.drop(columns=["datetime"], errors="ignore")


def latest_fetch_per_date(fetches: list[tuple[tuple, pd.DataFrame]]) -> pd.DataFrame:
    """Merge overlapping bronze fetches: for each transit_date, keep only the rows
    from the most recent fetch that contains it.

    Bronze windows overlap (every ingestion re-fetches a lookback window), and a
    later fetch of a date supersedes an earlier one — the earlier copy may have been
    taken before the MTA finished publishing that day. Each fetch comes with an
    orderable recency key (fetch time first); ranking on the full key means two
    fetches can never tie, so a date is always taken from exactly one of them.
    """
    ranked = sorted((f for f in fetches if not f[1].empty), key=lambda f: f[0])
    frames = [df.assign(_rank=rank) for rank, (_, df) in enumerate(ranked)]
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined["transit_date"] = pd.to_datetime(combined["transit_date"])
    newest = combined.groupby("transit_date")["_rank"].transform("max")
    return combined[combined["_rank"] == newest].drop(columns="_rank").reset_index(drop=True)


def combine_silver(df_old: pd.DataFrame | None, df_new: pd.DataFrame) -> pd.DataFrame:
    """Replace every date present in df_new, keep the rest of df_old, sort by date.

    Replace-by-date, not append: a re-fetched date must supersede the old copy. An
    append would keep both when a value was revised, double-counting that day once
    gold sums the stations.
    """
    if df_old is None:
        combined = df_new
    else:
        kept = df_old[~df_old["transit_date"].isin(df_new["transit_date"].unique())]
        combined = pd.concat([kept, df_new], ignore_index=True)
    return combined.sort_values("transit_date").reset_index(drop=True)


def run() -> None:
    logger.info("Starting silver merge")
    s3 = get_s3_client()

    if SILVER_LOCAL_PATH.exists():
        existing_dates = pd.read_parquet(SILVER_LOCAL_PATH, columns=["transit_date"])
        last_silver_date = pd.to_datetime(existing_dates["transit_date"]).max()
        # Reprocess the lookback window too: ingestion re-fetched it, and any date in
        # it may have been revised since silver last saw it.
        cutoff = last_silver_date - pd.Timedelta(days=MTA_LOOKBACK_DAYS)
        incremental = True
        logger.info(f"Existing silver found. Latest date: {last_silver_date.date()}, "
                    f"reprocessing dates after {cutoff.date()}")
    else:
        cutoff = pd.to_datetime("1900-01-01")
        incremental = False
        logger.info("No silver file found. Running full build.")

    all_mta = list_s3_files_with_mtime(s3, S3_MTA_PREFIX)
    mta_files = [(k, t) for k, t in all_mta if get_end_date_from_filename(k) > cutoff]

    if not mta_files:
        logger.info("No new MTA files to process.")
        return

    logger.info(f"Processing {len(mta_files)} MTA file(s)")
    # Recency: write time, then window end, then key — later fetches end later.
    fetches = [((t, get_end_date_from_filename(k), k), read_s3_csv(s3, k)) for k, t in mta_files]

    missing_mta = {c for _, df in fetches for c in _REQUIRED_MTA_COLS - set(df.columns)}
    if missing_mta:
        logger.error(f"MTA data missing required columns: {missing_mta} — aborting silver merge")
        return

    df_new_mta = latest_fetch_per_date(fetches)
    df_new_mta = df_new_mta[df_new_mta["transit_date"] > cutoff]

    if df_new_mta.empty:
        logger.info("All MTA records already in silver.")
        return

    all_weather_keys = list_s3_files(s3, S3_WEATHER_HIST_PREFIX)
    weather_keys = [k for k in all_weather_keys if get_end_date_from_filename(k) > cutoff]

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
    # Overlapping weather files would otherwise duplicate ridership rows in the join.
    df_weather = df_weather.drop_duplicates(subset="datetime", keep="last")

    relevant_dates = df_new_mta["transit_date"].unique()
    df_weather = df_weather[df_weather["datetime"].isin(relevant_dates)]

    df_merged = join_weather(df_new_mta, df_weather)

    if incremental:
        df_old = pd.read_parquet(SILVER_LOCAL_PATH)
        df_final = combine_silver(df_old, df_merged)
    else:
        df_final = combine_silver(None, df_merged)

    SILVER_LOCAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_parquet(SILVER_LOCAL_PATH, index=False, engine="pyarrow")

    logger.info(f"Silver updated: {len(df_final):,} rows, latest: {df_final['transit_date'].max().date()}")


if __name__ == "__main__":
    run()
