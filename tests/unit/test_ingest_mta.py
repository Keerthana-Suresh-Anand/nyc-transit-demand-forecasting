"""Tests for MTA ingestion: SoQL builder, fetch logic, and run() orchestration."""
from datetime import date
from unittest.mock import MagicMock, patch
from urllib.parse import unquote

import boto3
import pandas as pd
import pytest
from moto import mock_aws

from src.ingestion.ingest_mta import build_soql_query, fetch_mta_data, run
from src.utils.config import S3_MTA_PREFIX, S3_MTA_WATERMARK

BUCKET = "test-bucket"


def _decode(query: str) -> str:
    return unquote(query)


def test_soql_query_contains_date_range():
    q = _decode(build_soql_query(date(2025, 1, 1), date(2025, 1, 31), limit=1000, offset=0))
    assert "2025-01-01T00:00:00" in q
    assert "2025-01-31T23:59:59" in q


def test_soql_query_filters_subway_only():
    q = _decode(build_soql_query(date(2025, 1, 1), date(2025, 1, 7), limit=500, offset=0))
    assert "transit_mode = 'subway'" in q


def test_soql_query_groups_by_date_and_station():
    q = _decode(build_soql_query(date(2025, 1, 1), date(2025, 1, 7), limit=500, offset=0))
    assert "GROUP BY" in q
    assert "station_complex" in q


def test_soql_query_respects_limit_and_offset():
    q = _decode(build_soql_query(date(2025, 1, 1), date(2025, 1, 7), limit=250, offset=500))
    assert "LIMIT 250" in q
    assert "OFFSET 500" in q


def test_soql_query_zero_offset():
    q = _decode(build_soql_query(date(2025, 1, 1), date(2025, 1, 7), limit=500, offset=0))
    assert "OFFSET 0" in q


def _make_csv_response(rows: list[tuple]) -> MagicMock:
    header = "transit_date,station_complex,borough,daily_ridership"
    lines = [header] + [f"{r[0]},{r[1]},{r[2]},{r[3]}" for r in rows]
    mock_resp = MagicMock()
    mock_resp.content = "\n".join(lines).encode("utf-8")
    mock_resp.raise_for_status.return_value = None
    return mock_resp


class TestFetchMtaData:
    def test_returns_dataframe_with_expected_columns(self):
        resp = _make_csv_response([("2025-01-01", "Grand Central", "Manhattan", "50000")])
        with patch("src.ingestion.ingest_mta.requests.get", return_value=resp):
            df = fetch_mta_data(date(2025, 1, 1), date(2025, 1, 31))
        assert {"transit_date", "station_complex", "borough", "daily_ridership"}.issubset(df.columns)

    def test_returns_correct_row_count(self):
        rows = [(f"2025-01-{i+1:02d}", "Station A", "Manhattan", str(50000 + i)) for i in range(5)]
        resp = _make_csv_response(rows)
        with patch("src.ingestion.ingest_mta.requests.get", return_value=resp):
            df = fetch_mta_data(date(2025, 1, 1), date(2025, 1, 31))
        assert len(df) == 5

    def test_returns_empty_dataframe_when_response_is_empty(self):
        mock_resp = MagicMock()
        mock_resp.content = b""
        mock_resp.raise_for_status.return_value = None
        with patch("src.ingestion.ingest_mta.requests.get", return_value=mock_resp):
            df = fetch_mta_data(date(2025, 1, 1), date(2025, 1, 31))
        assert df.empty

    def test_raises_on_http_error(self):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = Exception("HTTP 500")
        with patch("src.ingestion.ingest_mta.requests.get", return_value=mock_resp):
            with pytest.raises(Exception):
                fetch_mta_data(date(2025, 1, 1), date(2025, 1, 31))

    def test_exits_when_app_token_missing(self, monkeypatch):
        monkeypatch.setattr("src.ingestion.ingest_mta.NY_APP_TOKEN", "")
        with pytest.raises(SystemExit):
            fetch_mta_data(date(2025, 1, 1), date(2025, 1, 31))


@pytest.fixture()
def s3():
    with mock_aws():
        client = boto3.client(
            "s3",
            region_name="us-east-1",
            aws_access_key_id="test-key",
            aws_secret_access_key="test-secret",
        )
        client.create_bucket(Bucket=BUCKET)
        yield client


class TestIngestMtaRun:
    def _sample_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "transit_date": ["2025-02-01", "2025-02-02"],
            "station_complex": ["Grand Central", "Times Sq"],
            "borough": ["Manhattan", "Manhattan"],
            "daily_ridership": [50000, 60000],
        })

    def test_skips_fetch_when_data_is_up_to_date(self, s3, monkeypatch):
        monkeypatch.setattr("src.ingestion.ingest_mta.get_s3_client", lambda: s3)
        # Watermark = today means start > end → skip
        s3.put_object(Bucket=BUCKET, Key=S3_MTA_WATERMARK, Body=str(date.today()).encode())

        fetch_called = []
        monkeypatch.setattr(
            "src.ingestion.ingest_mta.fetch_mta_data",
            lambda s, e: fetch_called.append(True) or pd.DataFrame(),
        )
        run()
        assert not fetch_called

    def test_writes_csv_to_s3_when_new_data_available(self, s3, monkeypatch):
        monkeypatch.setattr("src.ingestion.ingest_mta.get_s3_client", lambda: s3)
        # Old watermark → triggers fetch
        s3.put_object(Bucket=BUCKET, Key=S3_MTA_WATERMARK, Body=b"2025-01-01")
        monkeypatch.setattr("src.ingestion.ingest_mta.fetch_mta_data", lambda s, e: self._sample_df())

        run()

        from src.utils.s3_helpers import list_s3_files
        keys = list_s3_files(s3, S3_MTA_PREFIX, ".csv")
        assert len(keys) == 1

    def test_updates_watermark_after_successful_fetch(self, s3, monkeypatch):
        monkeypatch.setattr("src.ingestion.ingest_mta.get_s3_client", lambda: s3)
        s3.put_object(Bucket=BUCKET, Key=S3_MTA_WATERMARK, Body=b"2025-01-01")
        monkeypatch.setattr("src.ingestion.ingest_mta.fetch_mta_data", lambda s, e: self._sample_df())

        run()

        obj = s3.get_object(Bucket=BUCKET, Key=S3_MTA_WATERMARK)
        new_watermark = obj["Body"].read().decode().strip()
        assert new_watermark == "2025-02-02"

    def test_skips_write_when_fetch_returns_empty(self, s3, monkeypatch):
        monkeypatch.setattr("src.ingestion.ingest_mta.get_s3_client", lambda: s3)
        s3.put_object(Bucket=BUCKET, Key=S3_MTA_WATERMARK, Body=b"2025-01-01")
        monkeypatch.setattr("src.ingestion.ingest_mta.fetch_mta_data", lambda s, e: pd.DataFrame())

        run()

        from src.utils.s3_helpers import list_s3_files
        assert list_s3_files(s3, S3_MTA_PREFIX, ".csv") == []
