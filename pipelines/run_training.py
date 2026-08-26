"""Pipeline: train SARIMAX + XGBoost on the gold layer, evaluate and select champion.

Gold is built by the ingestion pipeline and downloaded from S3 by the workflow — this
pipeline is a pure consumer of the gold layer, it does not rebuild it.
"""
from datetime import date

from pipelines._runner import run_pipeline
from src.evaluation import evaluate_models, walk_forward
from src.training import train_sarimax, train_xgboost
from src.utils.config import (
    REPORTS_DIR,
    S3_RETRAIN_FLAG_KEY,
    S3_SARIMAX_COEF_KEY,
    S3_SHAP_KEY,
    S3_WALKFORWARD_KEY,
)
from src.utils.lineage import read_gold_dvc_md5
from src.utils.logger import get_logger
from src.utils.s3_helpers import delete_s3_key, get_s3_client, upload_s3_file, write_s3_json

logger = get_logger(__name__)


def _run() -> dict:
    gold_dvc_hash = read_gold_dvc_md5()
    if gold_dvc_hash:
        logger.info(f"Gold SARIMA DVC hash: {gold_dvc_hash}")

    train_sarimax.run(gold_dvc_hash=gold_dvc_hash)
    train_xgboost.run(gold_dvc_hash=gold_dvc_hash)
    champion = evaluate_models.run()
    logger.info(f"Champion model: {champion}")

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
    return {"champion_model": champion, "gold_dvc_hash": gold_dvc_hash}


def run() -> None:
    run_pipeline("training", _run)


if __name__ == "__main__":
    run()
