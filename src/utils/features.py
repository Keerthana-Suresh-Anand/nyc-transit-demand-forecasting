"""Shared feature definitions for the XGBoost model.

Categorical features use fixed category ranges so that training, evaluation,
and inference always agree on the encoding regardless of which values happen
to appear in a given data slice. This prevents train/serve skew when a single
forecast row (or a short evaluation window) does not contain every weekday or
month.
"""
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
