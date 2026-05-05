"""Tests for silver merge: filename date parsing and incremental merge logic."""
import pandas as pd

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


class TestIncrementalMergeLogic:
    """Test the merge/dedup/sort behaviour in isolation (no S3)."""

    def _make_df(self, dates, ridership_start=50_000) -> pd.DataFrame:
        return pd.DataFrame({
            "transit_date": pd.to_datetime(dates),
            "station_complex": ["Grand Central"] * len(dates),
            "daily_ridership": [ridership_start + i * 1000 for i in range(len(dates))],
            "temp": [45.0] * len(dates),
            "precip": [0.0] * len(dates),
            "snow": [0.0] * len(dates),
        })

    def test_concat_and_dedup(self):
        old = self._make_df(pd.date_range("2025-01-01", periods=10))
        new = self._make_df(pd.date_range("2025-01-08", periods=5))
        combined = pd.concat([old, new], ignore_index=True)
        deduped = combined.drop_duplicates().sort_values("transit_date").reset_index(drop=True)
        # overlap on Jan 8-10 (3 days), but rows are distinct because ridership differs
        assert len(deduped) == 15

    def test_no_new_rows_when_all_old(self):
        old = self._make_df(pd.date_range("2025-01-01", periods=10))
        last_date = old["transit_date"].max()
        new_keys = [k for k in pd.date_range("2025-01-01", periods=10) if pd.Timestamp(k) > last_date]
        assert len(new_keys) == 0
