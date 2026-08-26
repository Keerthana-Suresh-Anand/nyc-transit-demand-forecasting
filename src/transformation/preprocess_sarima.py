import pandas as pd

from src.utils.config import GOLD_SARIMA_LOCAL_PATH, SILVER_LOCAL_PATH
from src.utils.features import us_holidays_spanning
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Sanity bounds — loose on purpose: they catch upstream breakage (unit changes,
# decimal shifts, half-empty fetches), not ordinary variation.
_RIDERSHIP_BOUNDS = (100_000, 7_000_000)   # holiday lows ~1.2M, pre-COVID peaks ~5.7M
_TEMP_BOUNDS_C = (-30.0, 45.0)             # Central Park records: ~-26°C / ~41°C
_MAX_DAILY_PRECIP_MM = 400.0               # NYC daily record ~180mm


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """Silver station-level rows → daily city-wide gold frame. Pure — no I/O."""
    df_daily = (
        df.groupby("transit_date")
        .agg({"daily_ridership": "sum", "temp": "mean", "precip": "mean", "snow": "mean"})
        .sort_index()
    )
    df_daily = df_daily.asfreq("D")

    # The MTA series has been gap-free since 2022 — any interpolation is a signal
    # that ingestion needs a look.
    n_gaps = int(df_daily["daily_ridership"].isna().sum())
    if n_gaps:
        logger.warning(
            f"{n_gaps} missing ridership day(s) filled by linear interpolation — "
            "check ingestion for gaps"
        )
    df_daily["daily_ridership"] = df_daily["daily_ridership"].interpolate(method="linear")
    df_daily["temp"] = df_daily["temp"].interpolate(method="linear")
    df_daily[["precip", "snow"]] = df_daily[["precip", "snow"]].fillna(0)

    us_holidays = us_holidays_spanning(df_daily.index.min().year, df_daily.index.max().year)
    df_daily["is_holiday"] = df_daily.index.map(lambda x: 1 if x in us_holidays else 0)

    # Snow impacts ridership the following day (slush/cleanup effect)
    df_daily["snow_lag1"] = df_daily["snow"].shift(1).fillna(0)

    return df_daily


def validate_gold(df: pd.DataFrame) -> None:
    """Value-level sanity checks on the gold frame before it is persisted.

    Raises on structural problems (empty frame, nulls that survived the fill);
    warns loudly on out-of-range values so a bad upstream fetch is visible in the
    pipeline logs without blocking on a single anomalous day.
    """
    if df.empty:
        raise ValueError("Gold SARIMA frame is empty — refusing to persist")
    nulls = df[["daily_ridership", "temp", "precip", "snow"]].isna().sum()
    if nulls.any():
        raise ValueError(f"Gold SARIMA frame has nulls after fill: {nulls[nulls > 0].to_dict()}")

    checks = {
        "daily_ridership outside "
        f"[{_RIDERSHIP_BOUNDS[0]:,}, {_RIDERSHIP_BOUNDS[1]:,}]": ~df["daily_ridership"].between(*_RIDERSHIP_BOUNDS),
        f"temp outside [{_TEMP_BOUNDS_C[0]}, {_TEMP_BOUNDS_C[1]}]°C": ~df["temp"].between(*_TEMP_BOUNDS_C),
        f"precip outside [0, {_MAX_DAILY_PRECIP_MM}]mm": ~df["precip"].between(0, _MAX_DAILY_PRECIP_MM),
        f"snow outside [0, {_MAX_DAILY_PRECIP_MM}]mm": ~df["snow"].between(0, _MAX_DAILY_PRECIP_MM),
        "is_holiday not in {0, 1}": ~df["is_holiday"].isin([0, 1]),
    }
    for label, mask in checks.items():
        if mask.any():
            dates = df.index[mask][:5].strftime("%Y-%m-%d").tolist()
            logger.warning(f"Gold validation: {int(mask.sum())} day(s) with {label} (e.g. {dates})")


def run() -> pd.DataFrame:
    logger.info("Starting SARIMA preprocessing")

    df = pd.read_parquet(SILVER_LOCAL_PATH)
    df_daily = transform(df)
    validate_gold(df_daily)

    GOLD_SARIMA_LOCAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_daily.to_parquet(GOLD_SARIMA_LOCAL_PATH)

    logger.info(f"SARIMA gold saved: {len(df_daily)} days, {df_daily['is_holiday'].sum()} holidays flagged")
    return df_daily


if __name__ == "__main__":
    run()
