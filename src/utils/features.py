"""Shared feature definitions for the XGBoost model.

Categorical features use fixed category ranges so that training, evaluation,
and inference always agree on the encoding regardless of which values happen
to appear in a given data slice. This prevents train/serve skew when a single
forecast row (or a short evaluation window) does not contain every weekday or
month.
"""
import numpy as np
import pandas as pd

# Calendar features treated as native XGBoost categoricals (enable_categorical=True)
CATEGORICAL_FEATURES = ["day_of_week", "month"]

# Fixed, exhaustive category ranges
DOW_CATEGORIES = list(range(7))        # 0=Monday … 6=Sunday
MONTH_CATEGORIES = list(range(1, 13))  # 1=January … 12=December


def cast_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Cast calendar columns to pandas category dtype with fixed categories.

    Mutates and returns the same DataFrame. Values outside the fixed ranges
    become NaN, but day_of_week (0–6) and month (1–12) are always in range.
    """
    if "day_of_week" in df.columns:
        df["day_of_week"] = pd.Categorical(df["day_of_week"], categories=DOW_CATEGORIES)
    if "month" in df.columns:
        df["month"] = pd.Categorical(df["month"], categories=MONTH_CATEGORIES)
    return df


def iterative_xgb_predict(model, df: pd.DataFrame, start_idx: int, n_steps: int) -> np.ndarray:
    """Forecast ``n_steps`` days iteratively, propagating predicted ridership into
    the lag/rolling features — the same way ``generate_forecast`` serves production.

    Calendar and weather features use the actual values at each target row (known
    at forecast time), but ridership lags and rolling stats are filled with the
    model's own prior predictions rather than the true future values. This makes a
    holdout score reflect real 14-day inference instead of a one-shot fit that
    peeks at true lags. ``df`` must hold ``daily_ridership`` plus every feature
    column; predictions are returned in millions, matching the training target.
    """
    feature_cols = [c for c in df.columns if c != "daily_ridership"]
    ridership_lag_cols = {
        c: int(c.replace("ridership_lag", ""))
        for c in feature_cols if c.startswith("ridership_lag")
    }

    predictions: list[float] = []
    for step in range(n_steps):
        target_idx = start_idx + step
        target_row = df.iloc[target_idx]

        next_row: dict = {}
        for col in feature_cols:
            if col in ridership_lag_cols:
                lag = ridership_lag_cols[col]
                if len(predictions) >= lag:
                    next_row[col] = predictions[-lag]
                else:
                    next_row[col] = df["daily_ridership"].iloc[target_idx - lag] / 1_000_000
            elif col == "ridership_14d_avg":
                history = list(df["daily_ridership"].iloc[max(0, target_idx - 14):target_idx] / 1_000_000)
                window = (history + predictions)[-14:]
                next_row[col] = float(np.mean(window)) if window else 0.0
            elif col == "ridership_7d_std":
                history = list(df["daily_ridership"].iloc[max(0, target_idx - 14):target_idx] / 1_000_000)
                window = (history + predictions)[-7:]
                next_row[col] = float(np.std(window)) if len(window) >= 2 else 0.0
            elif col in CATEGORICAL_FEATURES:
                # Keep as int — casting through float then to a fixed integer
                # category range would turn the value into NaN.
                next_row[col] = int(target_row[col])
            else:
                next_row[col] = float(target_row[col])

        X_next = cast_categoricals(pd.DataFrame([next_row])[feature_cols])
        predictions.append(float(model.predict(X_next)[0]))

    return np.array(predictions)
