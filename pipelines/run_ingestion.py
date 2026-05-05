"""Pipeline: fetch new MTA + weather data and update silver layer."""
import sys
from datetime import date, datetime

from src.ingestion import ingest_events, ingest_mta, ingest_weather, merge_silver
from src.utils.config import S3_PIPELINE_RUNS_PREFIX, S3_SILVER_KEY, SILVER_LOCAL_PATH
from src.utils.logger import get_logger
from src.utils.s3_helpers import get_s3_client, upload_s3_file, write_s3_json

logger = get_logger(__name__)


def run() -> None:
    start = datetime.utcnow()
    status = "success"
    error_msg = None

    try:
        logger.info("=== Ingestion Pipeline START ===")
        ingest_mta.run()
        ingest_weather.run()
        ingest_events.run()
        merge_silver.run()
        s3 = get_s3_client()
        upload_s3_file(s3, SILVER_LOCAL_PATH, S3_SILVER_KEY)
        logger.info("=== Ingestion Pipeline COMPLETE ===")
    except Exception as e:
        status = "failure"
        error_msg = str(e)
        logger.error(f"Ingestion pipeline failed: {e}", exc_info=True)
    finally:
        duration = (datetime.utcnow() - start).total_seconds()
        log_entry = {
            "pipeline": "ingestion",
            "run_date": str(date.today()),
            "start_utc": start.isoformat(),
            "duration_seconds": duration,
            "status": status,
            "error": error_msg,
        }
        try:
            s3 = get_s3_client()
            write_s3_json(s3, log_entry, f"{S3_PIPELINE_RUNS_PREFIX}ingestion_{date.today()}.json")
        except Exception:
            pass

    if status == "failure":
        sys.exit(1)


if __name__ == "__main__":
    run()
