"""Pipeline: run performance monitoring and drift detection."""
from pipelines._runner import run_pipeline
from src.monitoring import monitor_performance
from src.utils.s3_helpers import get_s3_client


def _run() -> dict:
    s3 = get_s3_client()
    training_mae, baseline_source = monitor_performance.load_baseline_mae(s3)
    report = monitor_performance.run(training_mae=training_mae, baseline_source=baseline_source)
    return {"retrain_recommended": report.get("retrain_recommended")}


def run() -> None:
    run_pipeline("monitoring", _run)


if __name__ == "__main__":
    run()
