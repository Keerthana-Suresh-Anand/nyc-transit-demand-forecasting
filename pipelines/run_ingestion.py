"""Pipeline: fetch new MTA + weather data and update silver layer."""
import sys
from datetime import date, datetime

from src.ingestion import ingest_events, ingest_mta, ingest_weather, merge_silver
from src.transformation import preprocess_ml, preprocess_sarima
from src.utils.config import (
    GOLD_ML_LOCAL_PATH,
    GOLD_SARIMA_LOCAL_PATH,
    PIPELINE_IMAGE_DIGEST,
    S3_GOLD_ML_KEY,
    S3_GOLD_SARIMA_KEY,
    S3_PIPELINE_RUNS_PREFIX,
    S3_SILVER_KEY,
    SILVER_LOCAL_PATH,
)
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

        # Materialize the gold layer here — gold is a pure function of silver, so it
        # belongs where the data enters (weekly), not in the model pipelines. Training
        # and prediction are pure consumers of S3 gold. This keeps the dashboard's
        # actuals line and monitoring's rolling MAE fresh every week instead of only
        # when the monthly training run happens to rebuild gold.
        preprocess_sarima.run()
        preprocess_ml.run()

        s3 = get_s3_client()
        upload_s3_file(s3, SILVER_LOCAL_PATH, S3_SILVER_KEY)
        upload_s3_file(s3, GOLD_SARIMA_LOCAL_PATH, S3_GOLD_SARIMA_KEY)
        upload_s3_file(s3, GOLD_ML_LOCAL_PATH, S3_GOLD_ML_KEY)
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
            "image_digest": PIPELINE_IMAGE_DIGEST,
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
