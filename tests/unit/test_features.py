"""Tests for the shared feature builder — the train/serve-skew guarantee.

``iterative_xgb_predict`` is the single feature-construction path used by the
training holdout, the walk-forward backtest, and production serving. These tests
pin its recursive lag/rolling reconstruction against hand-computed values.
"""
import numpy as np
import pandas as pd
import pytest

from src.utils.features import (
    DOW_CATEGORIES,
    MONTH_CATEGORIES,
    cast_categoricals,
    iterative_xgb_predict,
    us_holidays_spanning,
)


class RecordingModel:
    """Stub model that records every feature row it is asked to predict on."""

    def __init__(self, outputs: list[float]):
        self.outputs = list(outputs)
        self.seen: list[pd.DataFrame] = []

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self.seen.append(X.copy())
        return np.array([self.outputs[len(self.seen) - 1]])


def _make_df(n_history: int = 20, n_future: int = 3) -> pd.DataFrame:
    """History rows carry actual ridership; future rows carry NaN placeholders."""
    n = n_history + n_future
    dates = pd.date_range("2025-01-01", periods=n, freq="D")
    ridership = [float((i + 1) * 1_000_000) for i in range(n_history)] + [np.nan] * n_future
    df = pd.DataFrame({
        "daily_ridership": ridership,
        "temp": [10.0 + i for i in range(n)],
        "day_of_week": dates.dayofweek,
        "month": dates.month,
        "ridership_lag1": np.nan,
        "ridership_lag7": np.nan,
        "ridership_14d_avg": np.nan,
        "ridership_7d_std": np.nan,
    }, index=dates)
    return df


class TestIterativeXgbPredict:
    def test_first_step_lags_come_from_actual_history(self):
        df = _make_df(n_history=20)
        model = RecordingModel([5.0, 6.0, 7.0])
        iterative_xgb_predict(model, df, start_idx=20, n_steps=3)

        first = model.seen[0].iloc[0]
        # History in millions: 1..20. lag1 = 20, lag7 = 14.
        assert first["ridership_lag1"] == pytest.approx(20.0)
        assert first["ridership_lag7"] == pytest.approx(14.0)
        assert first["ridership_14d_avg"] == pytest.approx(np.mean(range(7, 21)))
        assert first["ridership_7d_std"] == pytest.approx(np.std(range(14, 21), ddof=1))

    def test_predictions_feed_back_into_lags(self):
        df = _make_df(n_history=20)
        model = RecordingModel([5.0, 6.0, 7.0])
        preds = iterative_xgb_predict(model, df, start_idx=20, n_steps=3)

        # Step 2's lag1 is step 1's prediction; step 3's lag1 is step 2's.
        assert model.seen[1].iloc[0]["ridership_lag1"] == pytest.approx(5.0)
        assert model.seen[2].iloc[0]["ridership_lag1"] == pytest.approx(6.0)
        # Step 2's rolling mean includes the first prediction in its window.
        expected_avg = np.mean(list(range(8, 21)) + [5.0])
        assert model.seen[1].iloc[0]["ridership_14d_avg"] == pytest.approx(expected_avg)
        assert preds == pytest.approx([5.0, 6.0, 7.0])

    def test_never_reads_target_at_or_after_start_idx(self):
        df = _make_df(n_history=20)
        poisoned = df.copy()
        poisoned.loc[poisoned.index[20:], "daily_ridership"] = 999e6

        m_nan = RecordingModel([5.0, 6.0, 7.0])
        m_poisoned = RecordingModel([5.0, 6.0, 7.0])
        iterative_xgb_predict(m_nan, df, 20, 3)
        iterative_xgb_predict(m_poisoned, poisoned, 20, 3)

        # Identical feature rows either way — the poisoned target values at/after
        # start_idx must be invisible to feature construction.
        for a, b in zip(m_nan.seen, m_poisoned.seen, strict=True):
            pd.testing.assert_frame_equal(a, b)

    def test_calendar_and_weather_read_from_target_rows(self):
        df = _make_df(n_history=20)
        model = RecordingModel([5.0])
        iterative_xgb_predict(model, df, start_idx=20, n_steps=1)

        first = model.seen[0].iloc[0]
        target = df.iloc[20]
        assert first["temp"] == pytest.approx(target["temp"])
        assert first["day_of_week"] == target["day_of_week"]

    def test_categoricals_passed_with_fixed_dtype(self):
        df = _make_df(n_history=20)
        model = RecordingModel([5.0])
        iterative_xgb_predict(model, df, start_idx=20, n_steps=1)

        X = model.seen[0]
        assert list(X["day_of_week"].cat.categories) == DOW_CATEGORIES
        assert list(X["month"].cat.categories) == MONTH_CATEGORIES


class TestCastCategoricals:
    def test_fixed_categories_regardless_of_values_present(self):
        df = pd.DataFrame({"day_of_week": [0, 1], "month": [6, 6]})
        out = cast_categoricals(df)
        assert list(out["day_of_week"].cat.categories) == DOW_CATEGORIES
        assert list(out["month"].cat.categories) == MONTH_CATEGORIES

    def test_out_of_range_becomes_nan(self):
        df = pd.DataFrame({"day_of_week": [7], "month": [13]})
        out = cast_categoricals(df)
        assert out["day_of_week"].isna().all()
        assert out["month"].isna().all()

    def test_missing_columns_are_ignored(self):
        df = pd.DataFrame({"temp": [1.0]})
        assert "day_of_week" not in cast_categoricals(df).columns


class TestUsHolidaysSpanning:
    def test_covers_inclusive_range(self):
        cal = us_holidays_spanning(2022, 2026)
        assert pd.Timestamp("2022-07-04") in cal
        assert pd.Timestamp("2026-01-01") in cal

    def test_non_holiday_absent(self):
        cal = us_holidays_spanning(2025, 2025)
        assert pd.Timestamp("2025-01-06") not in cal
