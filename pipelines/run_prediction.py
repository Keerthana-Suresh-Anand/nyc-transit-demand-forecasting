"""Pipeline: generate a 14-day ensemble forecast from the gold layer.

Gold is built by the ingestion pipeline and downloaded from S3 by the workflow — this
pipeline is a pure consumer of the gold layer, it does not rebuild it.
"""
import sys
from datetime import date, datetime

from src.prediction import generate_forecast
from src.utils.config import PIPELINE_IMAGE_DIGEST, S3_PIPELINE_RUNS_PREFIX
from src.utils.logger import get_logger
from src.utils.s3_helpers import get_s3_client, write_s3_json

logger = get_logger(__name__)


def run() -> None:
    start = datetime.utcnow()
    status = "success"
    error_msg = None

    try:
        logger.info("=== Prediction Pipeline START ===")
        generate_forecast.run()
        logger.info("=== Prediction Pipeline COMPLETE ===")
    except Exception as e:
        status = "failure"
        error_msg = str(e)
        logger.error(f"Prediction pipeline failed: {e}", exc_info=True)
    finally:
        duration = (datetime.utcnow() - start).total_seconds()
        log_entry = {
            "pipeline": "prediction",
            "run_date": str(date.today()),
            "start_utc": start.isoformat(),
            "duration_seconds": duration,
            "status": status,
            "image_digest": PIPELINE_IMAGE_DIGEST,
            "error": error_msg,
        }
        try:
            s3 = get_s3_client()
            write_s3_json(s3, log_entry, f"{S3_PIPELINE_RUNS_PREFIX}prediction_{date.today()}.json")
        except Exception:
            pass

    if status == "failure":
        sys.exit(1)


if __name__ == "__main__":
    run()
