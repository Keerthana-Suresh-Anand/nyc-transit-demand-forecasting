"""Pipeline: generate a 14-day ensemble forecast from the gold layer.

Gold is built by the ingestion pipeline and downloaded from S3 by the workflow — this
pipeline is a pure consumer of the gold layer, it does not rebuild it.
"""
from pipelines._runner import run_pipeline
from src.prediction import generate_forecast


def run() -> None:
    run_pipeline("prediction", generate_forecast.run)


if __name__ == "__main__":
    run()
