"""Pipeline: run performance monitoring and drift detection."""
from pipelines._runner import run_pipeline
from src.monitoring import monitor_performance
from src.utils.config import S3_TRAINING_BASELINE_KEY
from src.utils.logger import get_logger
from src.utils.s3_helpers import get_s3_client, read_s3_json

logger = get_logger(__name__)


def _run() -> dict:
    s3 = get_s3_client()
    training_mae = None
    try:
        baseline = read_s3_json(s3, S3_TRAINING_BASELINE_KEY)
        training_mae = baseline.get("ensemble_mae")
        logger.info(f"Training baseline loaded — ensemble MAE: {training_mae:.4f}M")
    except Exception:
        logger.warning("No training baseline found — MAE threshold check disabled")

    report = monitor_performance.run(training_mae=training_mae)
    return {"retrain_recommended": report.get("retrain_recommended")}


def run() -> None:
    run_pipeline("monitoring", _run)


if __name__ == "__main__":
    run()
