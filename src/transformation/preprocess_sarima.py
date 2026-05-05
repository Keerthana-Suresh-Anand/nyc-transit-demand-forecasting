import holidays
import pandas as pd

from src.utils.config import SILVER_LOCAL_PATH, GOLD_SARIMA_LOCAL_PATH
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run() -> pd.DataFrame:
    logger.info("Starting SARIMA preprocessing")

    df = pd.read_parquet(SILVER_LOCAL_PATH)

    df_daily = (
        df.groupby("transit_date")
        .agg({"daily_ridership": "sum", "temp": "mean", "precip": "mean", "snow": "mean"})
        .sort_index()
    )
    df_daily = df_daily.asfreq("D")

    df_daily["daily_ridership"] = df_daily["daily_ridership"].interpolate(method="linear")
    df_daily["temp"] = df_daily["temp"].interpolate(method="linear")
    df_daily[["precip", "snow"]] = df_daily[["precip", "snow"]].fillna(0)

    us_holidays = holidays.US(years=[2024, 2025, 2026])
    df_daily["is_holiday"] = df_daily.index.map(lambda x: 1 if x in us_holidays else 0)

    # Snow impacts ridership the following day (slush/cleanup effect)
    df_daily["snow_lag1"] = df_daily["snow"].shift(1).fillna(0)

    GOLD_SARIMA_LOCAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_daily.to_parquet(GOLD_SARIMA_LOCAL_PATH)

    logger.info(f"SARIMA gold saved: {len(df_daily)} days, {df_daily['is_holiday'].sum()} holidays flagged")
    return df_daily


if __name__ == "__main__":
    run()
