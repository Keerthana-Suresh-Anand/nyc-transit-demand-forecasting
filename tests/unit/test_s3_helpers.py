"""Tests for S3 helper utilities."""
from datetime import date

import boto3
import pandas as pd
import pytest
from moto import mock_aws

import src.utils.s3_helpers as s3_helpers
from src.utils.s3_helpers import (
    MissingCredentialsError,
    download_s3_file,
    get_s3_client,
    list_s3_files,
    read_s3_csv,
    read_s3_json,
    read_s3_parquet,
    read_watermark,
    s3_key_exists,
    upload_s3_file,
    write_s3_csv,
    write_s3_json,
    write_s3_parquet,
    write_watermark,
)

BUCKET = "test-bucket"


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


class TestGetS3Client:
    def test_raises_when_credentials_missing(self, monkeypatch):
        for name in ("AWS_KEY", "AWS_SECRET", "AWS_REGION", "BUCKET"):
            monkeypatch.setattr(s3_helpers, name, None)
        with pytest.raises(MissingCredentialsError):
            get_s3_client()


class TestReadWatermark:
    def test_returns_none_when_key_missing(self, s3):
        assert read_watermark(s3, "no/such/key.txt") is None

    def test_returns_correct_date_when_key_exists(self, s3):
        s3.put_object(Bucket=BUCKET, Key="wm.txt", Body=b"2025-06-15")
        assert read_watermark(s3, "wm.txt") == date(2025, 6, 15)

    def test_returns_none_for_malformed_content(self, s3):
        s3.put_object(Bucket=BUCKET, Key="bad.txt", Body=b"not-a-date")
        assert read_watermark(s3, "bad.txt") is None


class TestWriteWatermark:
    def test_value_readable_after_write(self, s3):
        write_watermark(s3, "wm.txt", date(2025, 3, 10))
        obj = s3.get_object(Bucket=BUCKET, Key="wm.txt")
        assert obj["Body"].read().decode() == "2025-03-10"

    def test_second_write_overwrites_first(self, s3):
        write_watermark(s3, "wm.txt", date(2025, 1, 1))
        write_watermark(s3, "wm.txt", date(2025, 12, 31))
        assert read_watermark(s3, "wm.txt") == date(2025, 12, 31)


class TestListS3Files:
    def test_returns_matching_files(self, s3):
        s3.put_object(Bucket=BUCKET, Key="bronze/mta/a.csv", Body=b"x")
        s3.put_object(Bucket=BUCKET, Key="bronze/mta/b.csv", Body=b"y")
        result = list_s3_files(s3, "bronze/mta/", ".csv")
        assert sorted(result) == ["bronze/mta/a.csv", "bronze/mta/b.csv"]

    def test_filters_by_extension(self, s3):
        s3.put_object(Bucket=BUCKET, Key="bronze/mta/data.csv", Body=b"x")
        s3.put_object(Bucket=BUCKET, Key="bronze/mta/data.parquet", Body=b"y")
        result = list_s3_files(s3, "bronze/mta/", ".csv")
        assert result == ["bronze/mta/data.csv"]

    def test_returns_empty_for_missing_prefix(self, s3):
        assert list_s3_files(s3, "nonexistent/prefix/", ".csv") == []

    def test_respects_prefix_boundary(self, s3):
        s3.put_object(Bucket=BUCKET, Key="bronze/mta/f.csv", Body=b"x")
        s3.put_object(Bucket=BUCKET, Key="bronze/weather/f.csv", Body=b"y")
        result = list_s3_files(s3, "bronze/mta/", ".csv")
        assert result == ["bronze/mta/f.csv"]


class TestReadWriteCSV:
    def test_roundtrip(self, s3):
        df = pd.DataFrame({"col_a": [1, 2], "col_b": ["x", "y"]})
        write_s3_csv(s3, df, "test/data.csv")
        result = read_s3_csv(s3, "test/data.csv")
        pd.testing.assert_frame_equal(result, df)

    def test_multirow_df_preserved(self, s3):
        df = pd.DataFrame({"v": list(range(100))})
        write_s3_csv(s3, df, "test/large.csv")
        assert len(read_s3_csv(s3, "test/large.csv")) == 100


class TestReadWriteParquet:
    def test_roundtrip(self, s3):
        df = pd.DataFrame({"x": [1.0, 2.0], "y": ["a", "b"]})
        write_s3_parquet(s3, df, "test/data.parquet")
        result = read_s3_parquet(s3, "test/data.parquet")
        pd.testing.assert_frame_equal(result, df)

    def test_dtypes_preserved(self, s3):
        df = pd.DataFrame({"ints": [1, 2, 3], "floats": [1.1, 2.2, 3.3]})
        write_s3_parquet(s3, df, "test/typed.parquet")
        result = read_s3_parquet(s3, "test/typed.parquet")
        assert result.dtypes["ints"] == df.dtypes["ints"]
        assert result.dtypes["floats"] == df.dtypes["floats"]


class TestReadWriteJSON:
    def test_dict_roundtrip(self, s3):
        data = {"key": "value", "num": 42, "nested": {"a": 1}}
        write_s3_json(s3, data, "test/data.json")
        assert read_s3_json(s3, "test/data.json") == data

    def test_list_roundtrip(self, s3):
        data = [{"a": 1}, {"b": 2}]
        write_s3_json(s3, data, "test/list.json")
        assert read_s3_json(s3, "test/list.json") == data

    def test_content_type_is_json(self, s3):
        write_s3_json(s3, {"x": 1}, "test/ct.json")
        head = s3.head_object(Bucket=BUCKET, Key="test/ct.json")
        assert head["ContentType"] == "application/json"


class TestS3KeyExists:
    def test_true_for_existing_key(self, s3):
        s3.put_object(Bucket=BUCKET, Key="present.txt", Body=b"hi")
        assert s3_key_exists(s3, "present.txt") is True

    def test_false_for_missing_key(self, s3):
        assert s3_key_exists(s3, "absent.txt") is False


class TestUploadDownloadFile:
    def test_upload_then_download_roundtrip(self, s3, tmp_path):
        src = tmp_path / "input.bin"
        src.write_bytes(b"binary content")
        upload_s3_file(s3, src, "uploads/input.bin")
        dest = tmp_path / "output.bin"
        assert download_s3_file(s3, "uploads/input.bin", dest) is True
        assert dest.read_bytes() == b"binary content"

    def test_download_returns_false_for_missing_key(self, s3, tmp_path):
        dest = tmp_path / "out.txt"
        assert download_s3_file(s3, "does/not/exist.txt", dest) is False

    def test_download_creates_parent_directories(self, s3, tmp_path):
        s3.put_object(Bucket=BUCKET, Key="deep/nested/file.txt", Body=b"data")
        dest = tmp_path / "a" / "b" / "file.txt"
        download_s3_file(s3, "deep/nested/file.txt", dest)
        assert dest.exists()
