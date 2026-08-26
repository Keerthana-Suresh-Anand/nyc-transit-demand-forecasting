"""Tests for SARIMA preprocessing: holidays, snow_lag1 shift, gaps, validation."""
import pandas as pd
import pytest

from src.transformation.preprocess_sarima import transform, validate_gold


def _make_input(start="2025-01-01", periods=20) -> pd.DataFrame:
    dates = pd.date_range(start, periods=periods, freq="D")
    return pd.DataFrame({
        "transit_date": dates,
        "daily_ridership": [2_000_000.0] * periods,
        "temp": [35.0] * periods,
        "precip": [0.0] * periods,
        "snow": [0.0] * periods,
    })


class TestTransform:
    def test_new_years_day_flagged_as_holiday(self):
        result = transform(_make_input(start="2025-01-01", periods=5))
        assert result.loc["2025-01-01", "is_holiday"] == 1

    def test_holiday_flagged_across_all_data_years(self):
        # Regression: holiday years must come from the data, not a hardcoded list.
        result = transform(_make_input(start="2022-07-01", periods=10))
        assert result.loc["2022-07-04", "is_holiday"] == 1

    def test_non_holiday_not_flagged(self):
        result = transform(_make_input(start="2025-01-06", periods=5))
        # 2025-01-06 is a Monday, not a US holiday
        assert result.loc["2025-01-06", "is_holiday"] == 0

    def test_snow_lag1_shifts_by_one_day(self):
        df = _make_input(start="2025-01-06", periods=5)
        df.loc[df["transit_date"] == pd.Timestamp("2025-01-06"), "snow"] = 3.0
        result = transform(df)
        assert result.loc["2025-01-07", "snow_lag1"] == pytest.approx(3.0)
        assert result.loc["2025-01-06", "snow_lag1"] == pytest.approx(0.0)

    def test_gap_filled_by_interpolation(self):
        dates = pd.to_datetime(["2025-03-01", "2025-03-03"])  # gap on 03-02
        df = pd.DataFrame({
            "transit_date": dates,
            "daily_ridership": [2_000_000.0, 3_000_000.0],
            "temp": [40.0, 42.0],
            "precip": [0.0, 0.0],
            "snow": [0.0, 0.0],
        })
        result = transform(df)
        idx = pd.date_range("2025-03-01", "2025-03-03", freq="D")
        assert list(result.index) == list(idx)
        assert result.loc["2025-03-02", "daily_ridership"] == pytest.approx(2_500_000.0)

    def test_output_has_required_columns(self):
        result = transform(_make_input())
        for col in ["daily_ridership", "temp", "precip", "snow", "is_holiday", "snow_lag1"]:
            assert col in result.columns


class TestValidateGold:
    def test_valid_frame_passes(self):
        validate_gold(transform(_make_input()))

    def test_empty_frame_raises(self):
        with pytest.raises(ValueError, match="empty"):
            validate_gold(transform(_make_input()).iloc[0:0])

    def test_nulls_raise(self):
        gold = transform(_make_input())
        gold.loc[gold.index[0], "temp"] = None
        with pytest.raises(ValueError, match="nulls"):
            validate_gold(gold)

    def test_out_of_range_ridership_warns_not_raises(self):
        # The project logger sets propagate=False, so capture on it directly
        # instead of via caplog (which listens on the root logger).
        import logging

        from src.transformation import preprocess_sarima

        gold = transform(_make_input())
        gold.loc[gold.index[0], "daily_ridership"] = 50_000_000.0  # decimal-shift scenario

        records: list[logging.LogRecord] = []
        handler = logging.Handler()
        handler.emit = records.append
        preprocess_sarima.logger.addHandler(handler)
        try:
            validate_gold(gold)
        finally:
            preprocess_sarima.logger.removeHandler(handler)
        assert any("daily_ridership" in r.getMessage() for r in records)
