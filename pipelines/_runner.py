"""Shared pipeline runner: uniform logging, failure handling, and the S3 run-log entry."""
import sys
from collections.abc import Callable
from datetime import UTC, date, datetime

from src.utils.config import PIPELINE_IMAGE_DIGEST, S3_PIPELINE_RUNS_PREFIX
from src.utils.logger import get_logger
from src.utils.s3_helpers import get_s3_client, write_s3_json

logger = get_logger(__name__)


def run_pipeline(name: str, fn: Callable[[], dict | None]) -> None:
    """Run one pipeline entry point with uniform start/complete logging, failure
    capture, and a best-effort S3 run-log entry.

    ``fn`` may return a dict of extra fields to merge into the run-log entry
    (e.g. champion model, retrain recommendation). Exits non-zero on failure so
    the GitHub Actions job fails visibly.
    """
    start = datetime.now(UTC)
    status = "success"
    error_msg = None
    extra: dict = {}

    try:
        logger.info(f"=== {name.capitalize()} Pipeline START ===")
        extra = fn() or {}
        logger.info(f"=== {name.capitalize()} Pipeline COMPLETE ===")
    except Exception as e:
        status = "failure"
        error_msg = str(e)
        logger.error(f"{name.capitalize()} pipeline failed: {e}", exc_info=True)
    finally:
        log_entry = {
            "pipeline": name,
            "run_date": str(date.today()),
            "start_utc": start.isoformat(),
            "duration_seconds": (datetime.now(UTC) - start).total_seconds(),
            "status": status,
            **extra,
            "image_digest": PIPELINE_IMAGE_DIGEST,
            "error": error_msg,
        }
        try:
            s3 = get_s3_client()
            write_s3_json(s3, log_entry, f"{S3_PIPELINE_RUNS_PREFIX}{name}_{date.today()}.json")
        except Exception as e:
            logger.warning(f"Could not write pipeline run log: {e}")

    if status == "failure":
        sys.exit(1)
