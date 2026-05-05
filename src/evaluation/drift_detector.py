"""
Population Stability Index (PSI) drift detection + rolling MAE threshold check.
Writes a drift report to S3. If drift is critical, also writes a retrain flag.
"""
import json
from datetime import date

import numpy as np
import pandas as pd

from src.utils.config import (
    GOLD_SARIMA_LOCAL_PATH,
    MAE_RETRAIN_MULTIPLIER,
    PSI_CRITICAL_THRESHOLD,
    PSI_MODERATE_THRESHOLD,
    S3_DRIFT_REPORT_PREFIX,
    S3_RETRAIN_FLAG_KEY,
)
from src.utils.logger import get_logger
from src.utils.s3_helpers import get_s3_client, write_s3_json

logger = get_logger(__name__)

FEATURE_COLS = ["temp", "precip", "snow"]
REFERENCE_DAYS = 90   # baseline window for PSI
RECENT_DAYS = 14      # comparison window


def compute_psi(reference: np.ndarray, recent: np.ndarray, buckets: int = 10) -> float:
    """Compute Population Stability Index between reference and recent distributions."""
    breakpoints = np.percentile(reference, np.linspace(0, 100, buckets + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    ref_counts = np.histogram(reference, bins=breakpoints)[0]
    rec_counts = np.histogram(recent, bins=breakpoints)[0]

    ref_pct = (ref_counts + 1e-8) / len(reference)
    rec_pct = (rec_counts + 1e-8) / len(recent)

    psi = np.sum((rec_pct - ref_pct) * np.log(rec_pct / ref_pct))
    return float(psi)


def run(training_mae: float | None = None) -> dict:
    logger.info("Starting drift detection")
    s3 = get_s3_client()

    df = pd.read_parquet(GOLD_SARIMA_LOCAL_PATH)
    df.index = pd.to_datetime(df.index)

    if len(df) < REFERENCE_DAYS + RECENT_DAYS:
        logger.warning("Not enough data for drift detection.")
        return {"status": "insufficient_data"}

    reference = df.iloc[-(REFERENCE_DAYS + RECENT_DAYS):-RECENT_DAYS]
    recent = df.iloc[-RECENT_DAYS:]

    psi_scores = {}
    for col in FEATURE_COLS:
        if col in df.columns:
            psi_scores[col] = compute_psi(reference[col].values, recent[col].values)

    max_psi = max(psi_scores.values()) if psi_scores else 0.0
    psi_status = (
        "critical" if max_psi > PSI_CRITICAL_THRESHOLD
        else "moderate" if max_psi > PSI_MODERATE_THRESHOLD
        else "stable"
    )

    report = {
        "report_date": str(date.today()),
        "psi_scores": psi_scores,
        "max_psi": max_psi,
        "psi_status": psi_status,
        "retrain_recommended": psi_status == "critical",
    }

    if training_mae is not None:
        report["training_mae"] = training_mae
        report["mae_threshold"] = training_mae * MAE_RETRAIN_MULTIPLIER

    for col, psi in psi_scores.items():
        logger.info(f"PSI [{col}]: {psi:.4f} ({psi_status})")

    report_key = f"{S3_DRIFT_REPORT_PREFIX}drift_report_{date.today()}.json"
    write_s3_json(s3, report, report_key)

    if report["retrain_recommended"]:
        flag = {"trigger_date": str(date.today()), "reason": f"PSI={max_psi:.3f}", "max_psi": max_psi}
        write_s3_json(s3, flag, S3_RETRAIN_FLAG_KEY)
        logger.warning(f"Retrain flag written — PSI={max_psi:.3f} exceeds threshold {PSI_CRITICAL_THRESHOLD}")
    else:
        logger.info(f"No retrain needed — PSI={max_psi:.3f} (status: {psi_status})")

    return report


if __name__ == "__main__":
    run()
