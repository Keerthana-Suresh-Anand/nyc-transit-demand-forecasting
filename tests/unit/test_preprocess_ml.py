"""Tests for ML feature engineering: lag correctness and no future leakage.

These call the real `build_features` — an earlier version of this file
re-implemented the transformation inline and asserted against that copy, so it
stayed green no matter what the source did.
"""
import pandas as pd
import pytest

from src.transformation.preprocess_ml import RIDERSHIP_LAGS, build_features


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
    def test_expected_lag_columns_are_built(self):
        """The expected lags are spelled out here rather than read from
        RIDERSHIP_LAGS — a test that loops over the same constant the code uses
        can never detect a change to it."""
        result = build_features(_make_gold_sarima())
        expected = {f"ridership_lag{lag}" for lag in (1, 2, 3, 7, 14)}
        assert expected.issubset(result.columns)
        assert RIDERSHIP_LAGS == [1, 2, 3, 7, 14]

    def test_lag7_matches_value_7_rows_prior(self):
        df = _make_gold_sarima(periods=30)
        result = build_features(df)
        row = result.iloc[0]
        expected = df.loc[result.index[0] - pd.Timedelta(days=7), "daily_ridership"] / 1_000_000
        assert row["ridership_lag7"] == pytest.approx(expected)

    def test_lag1_matches_previous_day(self):
        df = _make_gold_sarima(periods=20)
        result = build_features(df)
        for i in range(len(result)):
            prev_date = result.index[i] - pd.Timedelta(days=1)
            if prev_date in df.index:
                expected = df.loc[prev_date, "daily_ridership"] / 1_000_000
                assert result.iloc[i]["ridership_lag1"] == pytest.approx(expected)

    def test_lags_are_in_millions(self):
        df = _make_gold_sarima()
        result = build_features(df)
        assert result["ridership_lag1"].max() < 100  # millions, not raw counts

    def test_rolling_stats_exclude_the_current_day(self):
        """The 14-day mean is shifted, so it must be computable from prior days only —
        checked by recomputing it from the source frame rather than trusting the column."""
        df = _make_gold_sarima(periods=40)
        result = build_features(df)
        idx = result.index[5]
        pos = df.index.get_loc(idx)
        expected = df["daily_ridership"].iloc[pos - 14:pos].mean() / 1_000_000
        assert result.loc[idx, "ridership_14d_avg"] == pytest.approx(expected)

    def test_weather_lags_match_previous_day(self):
        df = _make_gold_sarima(periods=30)
        result = build_features(df)
        idx = result.index[3]
        prev = idx - pd.Timedelta(days=1)
        assert result.loc[idx, "temp_lag1"] == pytest.approx(df.loc[prev, "temp"])
        assert result.loc[idx, "precip_lag1"] == pytest.approx(df.loc[prev, "precip"])

    def test_dropna_removes_lag_warmup_rows(self):
        df = _make_gold_sarima(periods=30)
        result = build_features(df)
        assert len(result) == len(df) - 14  # longest lag/rolling window
        assert result.isna().sum().sum() == 0

    def test_weekend_flag_correct(self):
        result = build_features(_make_gold_sarima())
        for idx in result.index:
            assert result.loc[idx, "is_weekend"] == int(idx.dayofweek >= 5)

    def test_calendar_features_written_as_ints(self):
        """Parquet does not preserve category dtype, so these must stay plain ints
        here and be cast on read (src/utils/features.cast_categoricals)."""
        result = build_features(_make_gold_sarima())
        assert result["day_of_week"].dtype.kind == "i"
        assert result["month"].dtype.kind == "i"

    def test_source_frame_not_mutated(self):
        df = _make_gold_sarima()
        before = df.columns.tolist()
        build_features(df)
        assert df.columns.tolist() == before
