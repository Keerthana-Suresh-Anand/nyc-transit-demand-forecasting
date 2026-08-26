"""Tests for the shared pipeline runner: run-log entry shape, failure handling."""
import json
from unittest.mock import MagicMock

import pytest

from pipelines._runner import run_pipeline


@pytest.fixture()
def s3(monkeypatch):
    client = MagicMock()
    monkeypatch.setattr("pipelines._runner.get_s3_client", lambda: client)
    return client


def _written_entry(s3) -> dict:
    (_, kwargs) = s3.put_object.call_args
    return json.loads(kwargs["Body"])


def test_success_writes_run_log_with_extra_fields(s3):
    run_pipeline("training", lambda: {"champion_model": "xgboost_production"})

    entry = _written_entry(s3)
    assert entry["pipeline"] == "training"
    assert entry["status"] == "success"
    assert entry["champion_model"] == "xgboost_production"
    assert entry["error"] is None
    assert "training_" in s3.put_object.call_args.kwargs["Key"]


def test_fn_returning_none_is_success(s3):
    run_pipeline("prediction", lambda: None)
    assert _written_entry(s3)["status"] == "success"


def test_failure_exits_nonzero_and_records_error(s3):
    def boom():
        raise RuntimeError("gold data missing")

    with pytest.raises(SystemExit) as exc:
        run_pipeline("ingestion", boom)

    assert exc.value.code == 1
    entry = _written_entry(s3)
    assert entry["status"] == "failure"
    assert "gold data missing" in entry["error"]


def test_run_log_write_failure_does_not_mask_success(s3):
    s3.put_object.side_effect = ConnectionError("s3 down")
    run_pipeline("monitoring", lambda: {})  # must not raise or exit
