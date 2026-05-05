"""Pipeline: preprocess latest data and generate 14-day ensemble forecast."""
import sys
from datetime import date, datetime

from src.prediction import generate_forecast
from src.transformation import preprocess_ml, preprocess_sarima
from src.utils.config import S3_PIPELINE_RUNS_PREFIX
from src.utils.logger import get_logger
from src.utils.s3_helpers import get_s3_client, write_s3_json

logger = get_logger(__name__)


def run() -> None:
    start = datetime.utcnow()
    status = "success"
    error_msg = None

    try:
        logger.info("=== Prediction Pipeline START ===")
        preprocess_sarima.run()
        preprocess_ml.run()
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
