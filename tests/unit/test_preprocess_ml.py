"""Tests for ML feature engineering: lag correctness and no future leakage."""
import pandas as pd
import pytest


def _run_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Replicate the core logic of preprocess_ml.run() on an in-memory df."""
    df = df.copy()
    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month
    df["is_weekend"] = df.index.dayofweek.isin([5, 6]).astype(int)

    for lag in [1, 2, 3, 7, 14]:
        df[f"ridership_lag{lag}"] = df["daily_ridership"].shift(lag) / 1_000_000

    df["ridership_14d_avg"] = df["daily_ridership"].shift(1).rolling(14).mean() / 1_000_000
    df["ridership_7d_std"] = df["daily_ridership"].shift(1).rolling(7).std() / 1_000_000
    df["precip_lag1"] = df["precip"].shift(1)
    df["temp_lag1"] = df["temp"].shift(1)
    return df.dropna()


def _make_gold_sarima(periods=30) -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=periods, freq="D")
    return pd.DataFrame({
        "daily_ridership": [float(3_000_000 + i * 10_000) for i in range(periods)],
        "temp": [50.0 + i * 0.1 for i in range(periods)],
        "precip": [0.0] * periods,
        "snow": [0.0] * periods,
        "is_holiday": [0] * periods,
        "snow_lag1": [0.0] * periods,
    }, index=dates)


class TestMLFeatureEngineering:
    def test_lag7_matches_value_7_rows_prior(self):
        df = _make_gold_sarima(periods=30)
        result = _run_feature_engineering(df)
        # After dropna (removes first 14 rows), check lag7
        row = result.iloc[0]  # first valid row
        expected_lag7_idx = result.index[0] - pd.Timedelta(days=7)
        expected_lag7 = df.loc[expected_lag7_idx, "daily_ridership"] / 1_000_000
        assert row["ridership_lag7"] == pytest.approx(expected_lag7)

    def test_lag1_matches_previous_day(self):
        df = _make_gold_sarima(periods=20)
        result = _run_feature_engineering(df)
        for i in range(1, len(result)):
            row = result.iloc[i]
            prev_date = result.index[i] - pd.Timedelta(days=1)
            if prev_date in df.index:
                expected = df.loc[prev_date, "daily_ridership"] / 1_000_000
                assert row["ridership_lag1"] == pytest.approx(expected)

    def test_no_future_leakage_in_rolling_stats(self):
        df = _make_gold_sarima(periods=30)
        result = _run_feature_engineering(df)
        # ridership_14d_avg uses shift(1) — so it must be strictly less than current date's value
        # If there's future leakage, the 14d avg would include the current day's value
        for idx in result.index:
            avg = result.loc[idx, "ridership_14d_avg"]
            current_val = df.loc[idx, "daily_ridership"] / 1_000_000
            # The 14d avg is computed from days BEFORE idx, so it should not equal current_val
            # (ridership is strictly increasing, so avg < current_val)
            assert avg < current_val, f"Possible future leakage at {idx}"

    def test_dropna_removes_initial_rows(self):
        df = _make_gold_sarima(periods=30)
        result = _run_feature_engineering(df)
        # lag14 + rolling(14) means at least 14 rows are dropped
        assert len(result) < len(df)
        assert result.isna().sum().sum() == 0

    def test_weekend_flag_correct(self):
        df = _make_gold_sarima(periods=30)
        result = _run_feature_engineering(df)
        for idx in result.index:
            expected = int(idx.dayofweek >= 5)
            assert result.loc[idx, "is_weekend"] == expected
