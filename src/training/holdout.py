"""Logging of the per-origin holdout forecasts both trainers produce.

The promotion gate reads the artifact back (``HOLDOUT_FORECASTS_ARTIFACT``): it is
the only out-of-sample evidence for a candidate, because the registered model is a
full-data refit.
"""
import mlflow
import numpy as np
import pandas as pd

from src.utils.config import HOLDOUT_FORECASTS_ARTIFACT, REPORTS_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)


def holdout_metrics(records: pd.DataFrame) -> dict[str, float]:
    err = records["y_pred"] - records["y_true"]
    return {
        "mae": float(err.abs().mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "mape": float((err.abs() / records["y_true"]).mean()),
        "bias": float(err.mean()),
    }


def log_holdout(records: pd.DataFrame, label: str) -> float:
    """Log metrics, a plot and the per-origin forecasts to the active MLflow run.
    Returns the holdout MAE."""
    import matplotlib.pyplot as plt

    metrics = holdout_metrics(records)
    mlflow.log_metrics({**metrics, "holdout_n_origins": records["origin"].nunique()})
    logger.info(
        f"{label} holdout ({records['origin'].nunique()} weekly 14-day forecasts, "
        f"{len(records)} points) — MAE {metrics['mae']:.4f}M  RMSE {metrics['rmse']:.4f}M  "
        f"MAPE {metrics['mape']:.2%}  bias {metrics['bias']:+.4f}M"
    )

    actual = records.drop_duplicates("date").sort_values("date")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(actual["date"], actual["y_true"], label="Actual", color="steelblue")
    for i, (_origin, grp) in enumerate(records.groupby("origin")):
        ax.plot(grp["date"], grp["y_pred"], linestyle="--", color="darkorange",
                alpha=0.5 + 0.5 * (i + 1) / records["origin"].nunique(),
                label=f"{label} forecast" if i == 0 else None)
    ax.set_title(f"{label} holdout | MAE {metrics['mae']:.3f}M | MAPE {metrics['mape']:.2%}")
    ax.legend()
    plot_path = REPORTS_DIR / f"{label.lower()}_training_forecast.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    mlflow.log_artifact(str(plot_path))

    out = records.assign(origin=records["origin"].dt.strftime("%Y-%m-%d"),
                         date=records["date"].dt.strftime("%Y-%m-%d"))
    path = REPORTS_DIR / HOLDOUT_FORECASTS_ARTIFACT
    out.to_json(path, orient="records")
    mlflow.log_artifact(str(path))
    return metrics["mae"]
