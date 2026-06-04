"""
Force a full silver rebuild by removing the local silver file so merge_silver
runs in full mode across all S3 bronze files (including the 2022-2024 backfill).
Uploads the rebuilt silver to S3 when done.
"""
import shutil

from src.ingestion import merge_silver
from src.utils.config import BUCKET, S3_SILVER_KEY, SILVER_LOCAL_PATH
from src.utils.logger import get_logger
from src.utils.s3_helpers import get_s3_client, upload_s3_file

logger = get_logger(__name__)


def run() -> None:
    if SILVER_LOCAL_PATH.exists():
        backup_path = SILVER_LOCAL_PATH.with_suffix(".parquet.bak")
        shutil.copy2(SILVER_LOCAL_PATH, backup_path)
        logger.info(f"Backed up existing silver to {backup_path}")
        SILVER_LOCAL_PATH.unlink()
        logger.info("Removed local silver file — merge will run in full-rebuild mode")
    else:
        logger.info("No local silver file found — running full build from scratch")

    merge_silver.run()

    s3 = get_s3_client()
    upload_s3_file(s3, SILVER_LOCAL_PATH, S3_SILVER_KEY)
    logger.info(f"Silver uploaded to s3://{BUCKET}/{S3_SILVER_KEY}")


if __name__ == "__main__":
    run()
