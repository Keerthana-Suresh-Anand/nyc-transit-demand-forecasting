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
    """Forecast ``n_steps`` days iteratively, feeding each predicted ridership back into
    the lag/rolling features.

    This is the single feature-construction path shared by the training holdout, the
    walk-forward backtest, and production serving (``generate_forecast``), so all three
    build features identically — no train/serve skew.

    Ridership-derived features (lags, 14-day mean, 7-day std) are computed from a running
    ``series`` of known-then-predicted ridership that grows one value per step. It never
    reads ``daily_ridership`` at or after ``start_idx``, so the same code works whether the
    target rows carry actuals (holdout) or are future placeholders (production). Calendar
    and weather features are read from each target row (known at forecast time). ``df`` must
    hold ``daily_ridership`` (actuals up to ``start_idx``) plus every feature column;
    predictions are returned in millions, matching the training target.
    """
    feature_cols = [c for c in df.columns if c != "daily_ridership"]
    ridership_lag_cols = {
        c: int(c.replace("ridership_lag", ""))
        for c in feature_cols if c.startswith("ridership_lag")
    }
    # Ridership known up to the day before the first forecast day, in millions. Grows by
    # one predicted value each step; every ridership-derived feature reads from here so
    # lags and rolling windows slide forward correctly with the forecast horizon.
    series = list(df["daily_ridership"].iloc[:start_idx].to_numpy() / 1_000_000)

    predictions: list[float] = []
    for step in range(n_steps):
        target_row = df.iloc[start_idx + step]

        next_row: dict = {}
        for col in feature_cols:
            if col in ridership_lag_cols:
                lag = ridership_lag_cols[col]
                next_row[col] = series[-lag] if len(series) >= lag else 0.0
            elif col == "ridership_14d_avg":
                window = series[-14:]
                next_row[col] = float(np.mean(window)) if window else 0.0
            elif col == "ridership_7d_std":
                window = series[-7:]
                # ddof=1 to match the pandas rolling std used to build the training features.
                next_row[col] = float(np.std(window, ddof=1)) if len(window) >= 2 else 0.0
            elif col in CATEGORICAL_FEATURES:
                # Keep as int — casting through float then to a fixed integer
                # category range would turn the value into NaN.
                next_row[col] = int(target_row[col])
            else:
                next_row[col] = float(target_row[col])

        X_next = cast_categoricals(pd.DataFrame([next_row])[feature_cols])
        pred = float(model.predict(X_next)[0])
        predictions.append(pred)
        series.append(pred)

    return np.array(predictions)
