"""Pipeline: preprocess data, train SARIMAX + XGBoost, evaluate and select champion."""
import sys
from datetime import date, datetime
from pathlib import Path

import yaml

from src.evaluation import evaluate_models
from src.training import train_sarimax, train_xgboost
from src.transformation import preprocess_ml, preprocess_sarima
from src.utils.config import (
    GOLD_ML_LOCAL_PATH,
    GOLD_SARIMA_LOCAL_PATH,
    REPORTS_DIR,
    S3_GOLD_ML_KEY,
    S3_GOLD_SARIMA_KEY,
    S3_PIPELINE_RUNS_PREFIX,
    S3_RETRAIN_FLAG_KEY,
    S3_SARIMAX_COEF_KEY,
    S3_SHAP_KEY,
)
from src.utils.logger import get_logger
from src.utils.s3_helpers import delete_s3_key, get_s3_client, upload_s3_file, write_s3_json

logger = get_logger(__name__)


def _read_silver_dvc_hash() -> str | None:
    dvc_file = Path("data/silver/mta_weather_merged.parquet.dvc")
    try:
        with open(dvc_file) as f:
            return yaml.safe_load(f)["outs"][0]["md5"]
    except Exception:
        return None


def run() -> None:
    start = datetime.utcnow()
    status = "success"
    error_msg = None
    champion = None

    silver_dvc_hash = _read_silver_dvc_hash()
    if silver_dvc_hash:
        logger.info(f"Silver DVC hash: {silver_dvc_hash}")

    try:
        logger.info("=== Training Pipeline START ===")
        preprocess_sarima.run()
        preprocess_ml.run()
        train_sarimax.run()
        train_xgboost.run()
        champion = evaluate_models.run()
        s3 = get_s3_client()
        upload_s3_file(s3, GOLD_SARIMA_LOCAL_PATH, S3_GOLD_SARIMA_KEY)
        upload_s3_file(s3, GOLD_ML_LOCAL_PATH, S3_GOLD_ML_KEY)
        shap_path = REPORTS_DIR / "xgboost_shap_summary.png"
        if shap_path.exists():
            upload_s3_file(s3, shap_path, S3_SHAP_KEY)
        coef_path = REPORTS_DIR / "sarimax_coefficients.json"
        if coef_path.exists():
            upload_s3_file(s3, coef_path, S3_SARIMAX_COEF_KEY)
        delete_s3_key(s3, S3_RETRAIN_FLAG_KEY)
        logger.info(f"=== Training Pipeline COMPLETE — champion: {champion} ===")
    except Exception as e:
        status = "failure"
        error_msg = str(e)
        logger.error(f"Training pipeline failed: {e}", exc_info=True)
    finally:
        duration = (datetime.utcnow() - start).total_seconds()
        log_entry = {
            "pipeline": "training",
            "run_date": str(date.today()),
            "start_utc": start.isoformat(),
            "duration_seconds": duration,
            "status": status,
            "champion_model": champion,
            "silver_dvc_hash": silver_dvc_hash,
            "error": error_msg,
        }
        try:
            s3 = get_s3_client()
            write_s3_json(s3, log_entry, f"{S3_PIPELINE_RUNS_PREFIX}training_{date.today()}.json")
        except Exception:
            pass

    if status == "failure":
        sys.exit(1)


if __name__ == "__main__":
    run()
