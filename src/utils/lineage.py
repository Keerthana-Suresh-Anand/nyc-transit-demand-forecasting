"""Data-lineage helpers shared by the pipelines.

The gold DVC pointer's md5 ties a trained model (MLflow tag) and a published
forecast (latest_forecast.json) to the exact data version they were built from.
"""
from pathlib import Path

import yaml

from src.utils.config import GOLD_SARIMA_LOCAL_PATH


def read_gold_dvc_md5() -> str | None:
    """md5 of the gold SARIMA parquet from its committed DVC pointer.

    Best-effort — returns None when the pointer is absent (e.g. local runs on a
    partial checkout). Anchored at the repo path from config, not the CWD.
    """
    dvc_file = Path(f"{GOLD_SARIMA_LOCAL_PATH}.dvc")
    try:
        with open(dvc_file) as f:
            return yaml.safe_load(f)["outs"][0]["md5"]
    except Exception:
        return None
