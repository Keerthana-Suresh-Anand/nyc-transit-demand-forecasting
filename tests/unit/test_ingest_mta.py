"""Tests for MTA ingestion: SoQL builder and date boundary logic."""
from datetime import date
from urllib.parse import unquote

import pytest

from src.ingestion.ingest_mta import build_soql_query


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
