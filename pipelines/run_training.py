"""Pipeline: train SARIMAX + XGBoost on the gold layer, evaluate and select champion.

Gold is built by the ingestion pipeline and downloaded from S3 by the workflow — this
pipeline is a pure consumer of the gold layer, it does not rebuild it.
"""
import sys
from datetime import date, datetime
from pathlib import Path

import yaml

from src.evaluation import evaluate_models, walk_forward
from src.training import train_sarimax, train_xgboost
from src.utils.config import (
    PIPELINE_IMAGE_DIGEST,
    REPORTS_DIR,
    S3_PIPELINE_RUNS_PREFIX,
    S3_RETRAIN_FLAG_KEY,
    S3_SARIMAX_COEF_KEY,
    S3_SHAP_KEY,
    S3_WALKFORWARD_KEY,
)
from src.utils.logger import get_logger
from src.utils.s3_helpers import delete_s3_key, get_s3_client, upload_s3_file, write_s3_json

logger = get_logger(__name__)


def _read_gold_dvc_hash() -> str | None:
    """md5 of the gold SARIMA parquet from its DVC pointer, tying the trained models to
    the exact gold version they saw. Gold is snapshotted to DVC by the ingestion pipeline.
    """
    dvc_file = Path("data/gold/mta_sarima.parquet.dvc")
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

    gold_dvc_hash = _read_gold_dvc_hash()
    if gold_dvc_hash:
        logger.info(f"Gold SARIMA DVC hash: {gold_dvc_hash}")

    try:
        logger.info("=== Training Pipeline START ===")
        train_sarimax.run()
        train_xgboost.run()
        champion = evaluate_models.run()
        s3 = get_s3_client()
        shap_path = REPORTS_DIR / "xgboost_shap_summary.png"
        if shap_path.exists():
            upload_s3_file(s3, shap_path, S3_SHAP_KEY)
        coef_path = REPORTS_DIR / "sarimax_coefficients.json"
        if coef_path.exists():
            upload_s3_file(s3, coef_path, S3_SARIMAX_COEF_KEY)

        # Recurring walk-forward backtest (robust, multi-origin) — supplementary to
        # the single-holdout champion gate above. Wrapped so a backtest failure can
        # never undo the model registration that already succeeded.
        try:
            wf_results = walk_forward.run()
            write_s3_json(s3, {"run_date": str(date.today()), **wf_results}, S3_WALKFORWARD_KEY)
            logger.info("Walk-forward evaluation written to S3")
        except Exception as e:
            logger.warning(f"Walk-forward evaluation skipped: {e}")

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
            "gold_dvc_hash": gold_dvc_hash,
            "image_digest": PIPELINE_IMAGE_DIGEST,
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
