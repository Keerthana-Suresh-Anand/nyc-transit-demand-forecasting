"""Tests for silver merge: filename date parsing and incremental merge logic."""
from datetime import UTC, datetime

import pandas as pd

from src.ingestion.merge_silver import combine_silver, join_weather, latest_fetch_per_date
from src.utils.s3_helpers import get_end_date_from_filename


class TestGetEndDateFromFilename:
    def test_standard_filename(self):
        ts = get_end_date_from_filename("bronze/mta/mta_daily_ridership_2025-01-01_2026-01-29.csv")
        assert ts == pd.Timestamp("2026-01-29")

    def test_filename_without_path(self):
        ts = get_end_date_from_filename("mta_daily_ridership_2025-06-01_2025-12-31.csv")
        assert ts == pd.Timestamp("2025-12-31")

    def test_malformed_filename_returns_sentinel(self):
        ts = get_end_date_from_filename("no_date_here.csv")
        assert ts == pd.Timestamp("1900-01-01")

    def test_empty_filename_returns_nat_or_sentinel(self):
        # pd.to_datetime("") returns NaT (not an exception), so the function returns NaT.
        # NaT comparisons evaluate to False, correctly excluding the file from processing.
        ts = get_end_date_from_filename("")
        assert pd.isna(ts) or ts == pd.Timestamp("1900-01-01")


class TestCombineSilver:
    """Append/dedupe/sort behaviour of the real `combine_silver` (no S3)."""

    def _make_df(self, dates, ridership_start=50_000) -> pd.DataFrame:
        return pd.DataFrame({
            "transit_date": pd.to_datetime(dates),
            "station_complex": ["Grand Central"] * len(dates),
            "daily_ridership": [ridership_start + i * 1000 for i in range(len(dates))],
            "temp": [45.0] * len(dates),
            "precip": [0.0] * len(dates),
            "snow": [0.0] * len(dates),
        })

    def test_full_build_returns_new_rows_sorted(self):
        new = self._make_df(pd.date_range("2025-01-05", periods=3)[::-1])  # reversed
        out = combine_silver(None, new)
        assert len(out) == 3
        assert out["transit_date"].is_monotonic_increasing

    def test_identical_rows_are_deduped(self):
        rows = self._make_df(pd.date_range("2025-01-01", periods=5))
        out = combine_silver(rows, rows.copy())  # same window fetched twice
        assert len(out) == 5

    def test_revised_dates_replace_old_rows(self):
        old = self._make_df(pd.date_range("2025-01-01", periods=10))
        new = self._make_df(pd.date_range("2025-01-08", periods=5), ridership_start=90_000)
        # Jan 8-10 overlap with different values: the new copy wins, no double count
        out = combine_silver(old, new)
        assert len(out) == 12  # Jan 1-12
        assert out["transit_date"].is_unique
        jan8 = out.loc[out["transit_date"] == "2025-01-08", "daily_ridership"].item()
        assert jan8 == 90_000

    def test_dates_outside_new_window_untouched(self):
        old = self._make_df(pd.date_range("2025-01-01", periods=10))
        new = self._make_df(pd.date_range("2025-01-08", periods=5), ridership_start=90_000)
        out = combine_silver(old, new)
        assert out.loc[out["transit_date"] == "2025-01-01", "daily_ridership"].item() == 50_000

    def test_result_index_is_reset(self):
        old = self._make_df(pd.date_range("2025-01-01", periods=4))
        new = self._make_df(pd.date_range("2025-02-01", periods=4))
        out = combine_silver(old, new)
        assert list(out.index) == list(range(len(out)))


class TestLatestFetchPerDate:
    """Overlapping bronze windows: the newest fetch of each date supersedes older ones."""

    def _fetch(self, dates, ridership):
        return pd.DataFrame({
            "transit_date": [str(d.date()) for d in dates],
            "station_complex": ["Grand Central"] * len(dates),
            "daily_ridership": [ridership] * len(dates),
        })

    def test_partial_day_replaced_by_later_fetch(self):
        # First fetch caught Jul 8 half-published; the lookback re-fetch has the full day
        first = self._fetch(pd.date_range("2026-07-02", "2026-07-08"), 100)
        refetch = self._fetch(pd.date_range("2026-07-02", "2026-07-15"), 400)
        out = latest_fetch_per_date([
            ((datetime(2026, 7, 14, tzinfo=UTC),), first),
            ((datetime(2026, 7, 21, tzinfo=UTC),), refetch),
        ])
        assert out["transit_date"].is_unique
        assert (out["daily_ridership"] == 400).all()
        assert len(out) == 14

    def test_order_of_inputs_does_not_matter(self):
        older = self._fetch(pd.date_range("2026-07-01", periods=3), 100)
        newer = self._fetch(pd.date_range("2026-07-02", periods=3), 400)
        a = latest_fetch_per_date([((datetime(2026, 7, 1, tzinfo=UTC),), older),
                                   ((datetime(2026, 7, 8, tzinfo=UTC),), newer)])
        b = latest_fetch_per_date([((datetime(2026, 7, 8, tzinfo=UTC),), newer),
                                   ((datetime(2026, 7, 1, tzinfo=UTC),), older)])
        key = ["transit_date", "daily_ridership"]
        assert a.sort_values(key)[key].reset_index(drop=True).equals(
            b.sort_values(key)[key].reset_index(drop=True))

    def test_dates_only_in_older_fetch_are_kept(self):
        older = self._fetch(pd.date_range("2026-07-01", periods=3), 100)
        newer = self._fetch(pd.date_range("2026-07-03", periods=3), 400)
        out = latest_fetch_per_date([((datetime(2026, 7, 1, tzinfo=UTC),), older),
                                     ((datetime(2026, 7, 8, tzinfo=UTC),), newer)])
        by_date = out.set_index("transit_date")["daily_ridership"]
        assert by_date[pd.Timestamp("2026-07-01")] == 100
        assert by_date[pd.Timestamp("2026-07-03")] == 400

    def test_all_stations_of_newest_fetch_kept(self):
        two_stations = pd.concat([
            self._fetch(pd.date_range("2026-07-08", periods=1), 400),
            self._fetch(pd.date_range("2026-07-08", periods=1), 500).assign(station_complex="Times Sq"),
        ], ignore_index=True)
        out = latest_fetch_per_date([
            ((datetime(2026, 7, 1, tzinfo=UTC),), self._fetch(pd.date_range("2026-07-08", periods=1), 1)),
            ((datetime(2026, 7, 8, tzinfo=UTC),), two_stations),
        ])
        assert sorted(out["daily_ridership"]) == [400, 500]

    def test_equal_fetch_times_never_keep_both_copies(self):
        # Same write time: the tie-breaker (window end) picks exactly one fetch per date
        t = datetime(2026, 7, 21, tzinfo=UTC)
        a = self._fetch(pd.date_range("2026-07-02", "2026-07-08"), 100)
        b = self._fetch(pd.date_range("2026-07-02", "2026-07-15"), 400)
        out = latest_fetch_per_date([((t, pd.Timestamp("2026-07-08")), a),
                                     ((t, pd.Timestamp("2026-07-15")), b)])
        assert out["transit_date"].is_unique
        assert (out["daily_ridership"] == 400).all()

    def test_empty_input(self):
        assert latest_fetch_per_date([]).empty


class TestJoinWeather:
    def _mta(self, dates):
        return pd.DataFrame({
            "transit_date": pd.to_datetime(dates),
            "station_complex": ["Grand Central"] * len(dates),
            "daily_ridership": [50_000] * len(dates),
        })

    def _weather(self, dates):
        return pd.DataFrame({
            "datetime": pd.to_datetime(dates),
            "temp": [45.0] * len(dates),
            "precip": [0.1] * len(dates),
            "snow": [0.0] * len(dates),
        })

    def test_weather_columns_attached(self):
        dates = pd.date_range("2025-01-01", periods=3)
        out = join_weather(self._mta(dates), self._weather(dates))
        assert {"temp", "precip", "snow"}.issubset(out.columns)
        assert out["temp"].notna().all()

    def test_join_key_column_dropped(self):
        dates = pd.date_range("2025-01-01", periods=3)
        out = join_weather(self._mta(dates), self._weather(dates))
        assert "datetime" not in out.columns

    def test_missing_weather_day_keeps_ridership_row(self):
        """Left join: a weather gap must never drop the ridership it belongs to."""
        mta_dates = pd.date_range("2025-01-01", periods=3)
        out = join_weather(self._mta(mta_dates), self._weather(mta_dates[:2]))
        assert len(out) == 3
        assert out["temp"].isna().sum() == 1

    def test_multiple_stations_per_day_all_get_weather(self):
        dates = pd.date_range("2025-01-01", periods=2)
        mta = pd.concat([self._mta(dates), self._mta(dates)], ignore_index=True)
        out = join_weather(mta, self._weather(dates))
        assert len(out) == 4
        assert out["temp"].notna().all()
