"""Pipeline: fetch new MTA + weather data and update silver layer."""
from pipelines._runner import run_pipeline
from src.ingestion import ingest_events, ingest_mta, ingest_weather, merge_silver
from src.transformation import preprocess_ml, preprocess_sarima
from src.utils.config import (
    GOLD_ML_LOCAL_PATH,
    GOLD_SARIMA_LOCAL_PATH,
    S3_GOLD_ML_KEY,
    S3_GOLD_SARIMA_KEY,
    S3_SILVER_KEY,
    SILVER_LOCAL_PATH,
)
from src.utils.s3_helpers import get_s3_client, upload_s3_file


def _run() -> None:
    ingest_mta.run()
    ingest_weather.run()
    ingest_events.run()
    merge_silver.run()

    # Materialize the gold layer here — gold is a pure function of silver, so it
    # belongs where the data enters (weekly), not in the model pipelines. Training
    # and prediction consume S3 gold, so building it on every ingestion keeps the
    # dashboard actuals and monitoring's rolling MAE fresh weekly.
    preprocess_sarima.run()
    preprocess_ml.run()

    s3 = get_s3_client()
    upload_s3_file(s3, SILVER_LOCAL_PATH, S3_SILVER_KEY)
    upload_s3_file(s3, GOLD_SARIMA_LOCAL_PATH, S3_GOLD_SARIMA_KEY)
    upload_s3_file(s3, GOLD_ML_LOCAL_PATH, S3_GOLD_ML_KEY)


def run() -> None:
    run_pipeline("ingestion", _run)


if __name__ == "__main__":
    run()
