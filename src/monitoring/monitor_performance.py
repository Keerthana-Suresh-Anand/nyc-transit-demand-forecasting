"""
Post-hoc performance monitor.

1. Loads the last N weekly forecast parquets from S3.
2. Matches predicted dates that are now in the past against gold actuals.
3. Computes rolling MAE / MAPE and compares against training-time MAE.
4. Runs PSI on the last 14 days of weather features vs 90-day reference window.
5. Writes a structured JSON report to S3.
6. If drift is critical OR rolling MAE exceeds threshold, writes retrain_flag.json.
"""
import io
from datetime import date

import numpy as np
import pandas as pd

from src.evaluation.drift_detector import compute_psi
from src.utils.config import (
    BUCKET,
    GOLD_SARIMA_LOCAL_PATH,
    MAE_RETRAIN_MULTIPLIER,
    PSI_CRITICAL_THRESHOLD,
    PSI_MODERATE_THRESHOLD,
    S3_DRIFT_REPORT_PREFIX,
    S3_FORECAST_PREFIX,
    S3_GOLD_SARIMA_KEY,
    S3_RETRAIN_FLAG_KEY,
)
from src.utils.logger import get_logger
from src.utils.s3_helpers import get_s3_client, list_s3_files, write_s3_json

logger = get_logger(__name__)

FEATURE_COLS = ["temp", "precip", "snow"]
REFERENCE_DAYS = 90
RECENT_DAYS = 14
N_FORECAST_FILES = 8   # how many past weekly forecasts to evaluate


def _load_gold(s3) -> pd.DataFrame | None:
    try:
        obj = s3.get_object(Bucket=BUCKET, Key=S3_GOLD_SARIMA_KEY)
        df = pd.read_parquet(io.BytesIO(obj["Body"].read()))
    except Exception:
        if GOLD_SARIMA_LOCAL_PATH.exists():
            df = pd.read_parquet(GOLD_SARIMA_LOCAL_PATH)
        else:
            return None
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def _load_past_forecasts(s3) -> pd.DataFrame | None:
    parquet_keys = sorted(list_s3_files(s3, S3_FORECAST_PREFIX, extension=".parquet"))
    if not parquet_keys:
        return None

    today = date.today()
    rows = []
    for key in parquet_keys[-N_FORECAST_FILES:]:
        try:
            obj = s3.get_object(Bucket=BUCKET, Key=key)
            df_fc = pd.read_parquet(io.BytesIO(obj["Body"].read()))
            df_fc["date"] = pd.to_datetime(df_fc["date"]).dt.date
            rows.append(df_fc[df_fc["date"] < today])
        except Exception as e:
            logger.warning(f"Could not load forecast file {key}: {e}")

    return pd.concat(rows, ignore_index=True) if rows else None


def _compute_forecast_metrics(
    past_fc: pd.DataFrame, gold: pd.DataFrame
) -> dict:
    records = []
    for _, row in past_fc.iterrows():
        ts = pd.Timestamp(row["date"])
        if ts in gold.index:
            actual_M = gold.loc[ts, "daily_ridership"] / 1_000_000
            pred_M = row["ensemble_forecast_M"]
            records.append({"actual_M": actual_M, "pred_M": pred_M,
                            "error_M": pred_M - actual_M})

    if not records:
        return {"n_evaluated": 0}

    errors = np.array([r["error_M"] for r in records])
    actuals = np.array([r["actual_M"] for r in records])
    mae = float(np.mean(np.abs(errors)))
    mape = float(np.mean(np.abs(errors / actuals)) * 100)
    return {
        "n_evaluated": len(records),
        "rolling_mae_M": mae,
        "rolling_mape_pct": mape,
    }


def _compute_psi_scores(gold: pd.DataFrame) -> dict[str, float]:
    if len(gold) < REFERENCE_DAYS + RECENT_DAYS:
        logger.warning("Not enough data for PSI — skipping")
        return {}

    reference = gold.iloc[-(REFERENCE_DAYS + RECENT_DAYS):-RECENT_DAYS]
    recent = gold.iloc[-RECENT_DAYS:]
    return {
        col: compute_psi(reference[col].values, recent[col].values)
        for col in FEATURE_COLS
        if col in gold.columns
    }


def run(training_mae: float | None = None) -> dict:
    logger.info("Starting performance monitoring")
    s3 = get_s3_client()
    today = date.today()

    gold = _load_gold(s3)
    if gold is None:
        logger.error("Could not load gold data — aborting monitoring")
        return {"status": "error", "reason": "no_gold_data"}

    # ── Forecast accuracy ──────────────────────────────────────────────────
    past_fc = _load_past_forecasts(s3)
    if past_fc is not None:
        fc_metrics = _compute_forecast_metrics(past_fc, gold)
    else:
        fc_metrics = {"n_evaluated": 0}
        logger.info("No past forecast files found")

    # ── PSI drift ─────────────────────────────────────────────────────────
    psi_scores = _compute_psi_scores(gold)
    max_psi = max(psi_scores.values(), default=0.0)
    psi_status = (
        "critical" if max_psi > PSI_CRITICAL_THRESHOLD
        else "moderate" if max_psi > PSI_MODERATE_THRESHOLD
        else "stable"
    )
    for col, psi in psi_scores.items():
        logger.info(f"PSI [{col}]: {psi:.4f}")

    # ── Retrain decision (MAE only — PSI is informational) ───────────────
    # PSI on weather features fires seasonal false alarms every spring/fall.
    # Retraining is triggered only by MAE degradation, never by PSI alone.
    retrain_reasons: list[str] = []
    rolling_mae = fc_metrics.get("rolling_mae_M")
    if training_mae is not None and rolling_mae is not None:
        threshold = training_mae * MAE_RETRAIN_MULTIPLIER
        if rolling_mae > threshold:
            retrain_reasons.append(
                f"Rolling MAE={rolling_mae:.4f}M exceeds {MAE_RETRAIN_MULTIPLIER}× training MAE={training_mae:.4f}M"
            )

    retrain_recommended = len(retrain_reasons) > 0

    # ── Build report ──────────────────────────────────────────────────────
    report = {
        "report_date": str(today),
        "psi_scores": psi_scores,
        "max_psi": max_psi,
        "psi_status": psi_status,
        "retrain_recommended": retrain_recommended,
        "retrain_reasons": retrain_reasons,
        **fc_metrics,
    }
    if training_mae is not None:
        report["training_mae_M"] = training_mae

    report_key = f"{S3_DRIFT_REPORT_PREFIX}drift_report_{today}.json"
    write_s3_json(s3, report, report_key)
    logger.info(f"Drift report written to {report_key}")

    if retrain_recommended:
        flag = {
            "trigger_date": str(today),
            "reasons": retrain_reasons,
            "max_psi": max_psi,
            "rolling_mae_M": rolling_mae,
        }
        write_s3_json(s3, flag, S3_RETRAIN_FLAG_KEY)
        logger.warning(f"Retrain flag written: {retrain_reasons}")
    else:
        logger.info(f"No retrain needed — PSI={max_psi:.3f} ({psi_status}), "
                    f"n_evaluated={fc_metrics.get('n_evaluated', 0)}")

    return report


if __name__ == "__main__":
    run()
