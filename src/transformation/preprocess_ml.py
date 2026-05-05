import pandas as pd

from src.utils.config import GOLD_SARIMA_LOCAL_PATH, GOLD_ML_LOCAL_PATH
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run() -> pd.DataFrame:
    logger.info("Starting ML feature engineering")

    df = pd.read_parquet(GOLD_SARIMA_LOCAL_PATH)
    df.index = pd.to_datetime(df.index)

    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month
    df["is_weekend"] = df.index.dayofweek.isin([5, 6]).astype(int)

    for lag in [1, 2, 3, 7, 14]:
        df[f"ridership_lag{lag}"] = df["daily_ridership"].shift(lag) / 1_000_000

    df["ridership_14d_avg"] = df["daily_ridership"].shift(1).rolling(14).mean() / 1_000_000
    df["ridership_7d_std"] = df["daily_ridership"].shift(1).rolling(7).std() / 1_000_000
    df["precip_lag1"] = df["precip"].shift(1)
    df["temp_lag1"] = df["temp"].shift(1)

    rows_before = len(df)
    df = df.dropna()
    logger.info(f"Dropped {rows_before - len(df)} rows due to NaN from lag/rolling operations")

    GOLD_ML_LOCAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(GOLD_ML_LOCAL_PATH)

    logger.info(f"ML gold saved: {len(df)} rows, {len(df.columns)} features")
    return df


if __name__ == "__main__":
    run()
