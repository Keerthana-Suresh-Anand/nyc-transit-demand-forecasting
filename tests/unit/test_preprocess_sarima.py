"""Tests for SARIMA preprocessing: holidays, snow_lag1 shift, no date gaps."""
import pandas as pd
import pytest
import holidays


class TestSarimaPreprocessing:
    """Test the transformation logic in isolation without disk I/O."""

    def _run_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Replicate the core logic of preprocess_sarima.run() on an in-memory df."""
        df = df.copy()
        df_daily = (
            df.groupby("transit_date")
            .agg({"daily_ridership": "sum", "temp": "mean", "precip": "mean", "snow": "mean"})
            .sort_index()
        )
        df_daily = df_daily.asfreq("D")
        df_daily["daily_ridership"] = df_daily["daily_ridership"].interpolate(method="linear")
        df_daily["temp"] = df_daily["temp"].interpolate(method="linear")
        df_daily[["precip", "snow"]] = df_daily[["precip", "snow"]].fillna(0)

        us_holidays = holidays.US(years=[2025, 2026])
        df_daily["is_holiday"] = df_daily.index.map(lambda x: 1 if x in us_holidays else 0)
        df_daily["snow_lag1"] = df_daily["snow"].shift(1).fillna(0)
        return df_daily

    def _make_input(self, start="2025-01-01", periods=20) -> pd.DataFrame:
        dates = pd.date_range(start, periods=periods, freq="D")
        return pd.DataFrame({
            "transit_date": dates,
            "daily_ridership": [2_000_000.0] * periods,
            "temp": [35.0] * periods,
            "precip": [0.0] * periods,
            "snow": [0.0] * periods,
        })

    def test_new_years_day_flagged_as_holiday(self):
        df = self._make_input(start="2025-01-01", periods=5)
        result = self._run_transform(df)
        assert result.loc["2025-01-01", "is_holiday"] == 1

    def test_non_holiday_not_flagged(self):
        df = self._make_input(start="2025-01-06", periods=5)
        result = self._run_transform(df)
        # 2025-01-06 is a Monday, not a US holiday
        assert result.loc["2025-01-06", "is_holiday"] == 0

    def test_snow_lag1_shifts_by_one_day(self):
        df = self._make_input(start="2025-01-06", periods=5)
        df.loc[df["transit_date"] == pd.Timestamp("2025-01-06"), "snow"] = 3.0
        result = self._run_transform(df)
        assert result.loc["2025-01-07", "snow_lag1"] == pytest.approx(3.0)
        assert result.loc["2025-01-06", "snow_lag1"] == pytest.approx(0.0)

    def test_no_date_gaps_after_asfreq(self):
        # Input with a missing day
        dates = pd.to_datetime(["2025-03-01", "2025-03-03"])  # gap on 03-02
        df = pd.DataFrame({
            "transit_date": dates,
            "daily_ridership": [2_000_000.0, 2_000_000.0],
            "temp": [40.0, 42.0],
            "precip": [0.0, 0.0],
            "snow": [0.0, 0.0],
        })
        result = self._run_transform(df)
        idx = pd.date_range("2025-03-01", "2025-03-03", freq="D")
        assert list(result.index) == list(idx)

    def test_output_has_required_columns(self):
        df = self._make_input()
        result = self._run_transform(df)
        for col in ["daily_ridership", "temp", "precip", "snow", "is_holiday", "snow_lag1"]:
            assert col in result.columns
