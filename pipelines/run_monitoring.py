"""Pipeline: run performance monitoring and drift detection."""
import sys
from datetime import date, datetime

from src.monitoring import monitor_performance
from src.utils.config import S3_PIPELINE_RUNS_PREFIX, S3_TRAINING_BASELINE_KEY
from src.utils.logger import get_logger
from src.utils.s3_helpers import get_s3_client, read_s3_json, write_s3_json

logger = get_logger(__name__)


def run() -> None:
    start = datetime.utcnow()
    status = "success"
    error_msg = None
    report = None

    try:
        logger.info("=== Monitoring Pipeline START ===")

        s3 = get_s3_client()
        training_mae = None
        try:
            baseline = read_s3_json(s3, S3_TRAINING_BASELINE_KEY)
            training_mae = baseline.get("ensemble_mae")
            logger.info(f"Training baseline loaded — ensemble MAE: {training_mae:.4f}M")
        except Exception:
            logger.warning("No training baseline found — MAE threshold check disabled")

        report = monitor_performance.run(training_mae=training_mae)
        logger.info("=== Monitoring Pipeline COMPLETE ===")
    except Exception as e:
        status = "failure"
        error_msg = str(e)
        logger.error(f"Monitoring pipeline failed: {e}", exc_info=True)
    finally:
        duration = (datetime.utcnow() - start).total_seconds()
        log_entry = {
            "pipeline": "monitoring",
            "run_date": str(date.today()),
            "start_utc": start.isoformat(),
            "duration_seconds": duration,
            "status": status,
            "retrain_recommended": report.get("retrain_recommended") if report else None,
            "error": error_msg,
        }
        try:
            s3 = get_s3_client()
            write_s3_json(s3, log_entry, f"{S3_PIPELINE_RUNS_PREFIX}monitoring_{date.today()}.json")
        except Exception:
            pass

    if status == "failure":
        sys.exit(1)


if __name__ == "__main__":
    run()
