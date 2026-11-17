"""Helpers for rows read back from the stored forecast parquets.

Pandas-only on purpose: imported by both monitoring and the dashboard.
"""
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

ORIGIN_COL = "_origin"


def drop_reforecasts(fc: pd.DataFrame) -> pd.DataFrame:
    """Keep one row per (origin, date, model versions); drops ``ORIGIN_COL``.

    Callers tag each file's rows with its origin (the file's first forecast date,
    i.e. last actual + 1) in ``ORIGIN_COL`` and concatenate files oldest first.

    Re-running prediction without new actuals (a manual dispatch, or a training run
    that promotes nothing) writes another file with the same origin and models —
    the same forecast again. Counting it twice inflated ``trigger_n`` and fired the
    retrain trigger on 7 unique days on 2026-11-16. Forecasts of one date from
    different origins (different lead times) or different model versions are
    distinct and all kept. The newest file wins a tie.
    """
    keys = [ORIGIN_COL, "date"] + [
        c for c in ("sarimax_version", "xgboost_version") if c in fc.columns
    ]
    n_before = len(fc)
    fc = fc.drop_duplicates(subset=keys, keep="last")
    if len(fc) < n_before:
        logger.info(f"Dropped {n_before - len(fc)} re-forecast rows (same origin + models)")
    return fc.drop(columns=ORIGIN_COL).reset_index(drop=True)
