import io
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import boto3
import pandas as pd

from src.utils.config import AWS_KEY, AWS_REGION, AWS_SECRET, BUCKET
from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_s3_client():
    missing = [k for k, v in {"AWS_ACCESS_KEY": AWS_KEY, "AWS_SECRET_KEY": AWS_SECRET,
                               "AWS_REGION": AWS_REGION, "AWS_BUCKET_NAME": BUCKET}.items() if not v]
    if missing:
        logger.error(f"Missing environment variables: {', '.join(missing)}")
        sys.exit(1)
    return boto3.client("s3", aws_access_key_id=AWS_KEY, aws_secret_access_key=AWS_SECRET,
                        region_name=AWS_REGION)


def read_watermark(s3_client, key: str) -> Optional[date]:
    """Read a date watermark from S3. Returns None if not found."""
    try:
        obj = s3_client.get_object(Bucket=BUCKET, Key=key)
        return datetime.strptime(obj["Body"].read().decode().strip(), "%Y-%m-%d").date()
    except s3_client.exceptions.NoSuchKey:
        return None
    except Exception as e:
        logger.warning(f"Could not read watermark {key}: {e}")
        return None


def write_watermark(s3_client, key: str, value: date) -> None:
    """Write a date watermark to S3."""
    s3_client.put_object(Bucket=BUCKET, Key=key, Body=str(value).encode())
    logger.debug(f"Watermark updated: {key} = {value}")


def list_s3_files(s3_client, prefix: str, extension: str = ".csv") -> list[str]:
    """List all S3 objects under prefix with the given extension."""
    paginator = s3_client.get_paginator("list_objects_v2")
    files = []
    for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(extension):
                files.append(obj["Key"])
    return files


def read_s3_csv(s3_client, key: str) -> pd.DataFrame:
    """Download a CSV from S3 and return as DataFrame."""
    logger.debug(f"Downloading: s3://{BUCKET}/{key}")
    obj = s3_client.get_object(Bucket=BUCKET, Key=key)
    return pd.read_csv(io.StringIO(obj["Body"].read().decode("utf-8")), low_memory=False)


def write_s3_csv(s3_client, df: pd.DataFrame, key: str) -> None:
    """Upload a DataFrame to S3 as CSV."""
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    s3_client.put_object(Bucket=BUCKET, Key=key, Body=buffer.getvalue())
    logger.info(f"Uploaded: s3://{BUCKET}/{key}")


def read_s3_parquet(s3_client, key: str) -> pd.DataFrame:
    """Download a Parquet file from S3 and return as DataFrame."""
    logger.debug(f"Downloading: s3://{BUCKET}/{key}")
    obj = s3_client.get_object(Bucket=BUCKET, Key=key)
    return pd.read_parquet(io.BytesIO(obj["Body"].read()))


def write_s3_parquet(s3_client, df: pd.DataFrame, key: str) -> None:
    """Upload a DataFrame to S3 as Parquet."""
    buffer = io.BytesIO()
    df.to_parquet(buffer, index=False, engine="pyarrow")
    buffer.seek(0)
    s3_client.put_object(Bucket=BUCKET, Key=key, Body=buffer.getvalue())
    logger.info(f"Uploaded: s3://{BUCKET}/{key}")


def write_s3_json(s3_client, data: dict | list, key: str) -> None:
    """Upload a dict or list to S3 as JSON."""
    import json
    body = json.dumps(data, indent=2, default=str)
    s3_client.put_object(Bucket=BUCKET, Key=key, Body=body.encode("utf-8"),
                         ContentType="application/json")
    logger.info(f"Uploaded: s3://{BUCKET}/{key}")


def read_s3_json(s3_client, key: str) -> dict | list:
    """Download a JSON file from S3."""
    import json
    obj = s3_client.get_object(Bucket=BUCKET, Key=key)
    return json.loads(obj["Body"].read().decode("utf-8"))


def s3_key_exists(s3_client, key: str) -> bool:
    """Check if an S3 key exists."""
    try:
        s3_client.head_object(Bucket=BUCKET, Key=key)
        return True
    except Exception:
        return False


def upload_s3_file(s3_client, local_path: Path, key: str) -> None:
    """Upload a local file to S3 by path (streams without loading into memory)."""
    s3_client.upload_file(str(local_path), BUCKET, key)
    logger.info(f"Uploaded: s3://{BUCKET}/{key}")


def download_s3_file(s3_client, key: str, local_path: Path) -> bool:
    """Download an S3 object to a local path. Returns False if key does not exist."""
    try:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        s3_client.download_file(BUCKET, key, str(local_path))
        logger.info(f"Downloaded: s3://{BUCKET}/{key} -> {local_path}")
        return True
    except s3_client.exceptions.ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
            logger.info(f"Not found in S3: {key}")
            return False
        raise


def get_end_date_from_filename(filename: str) -> pd.Timestamp:
    """Extract end date from filenames like mta_daily_ridership_2025-01-01_2026-01-29.csv"""
    try:
        clean_name = filename.replace(".csv", "").split("/")[-1]
        end_date_str = clean_name.split("_")[-1]
        return pd.to_datetime(end_date_str)
    except Exception:
        return pd.to_datetime("1900-01-01")
