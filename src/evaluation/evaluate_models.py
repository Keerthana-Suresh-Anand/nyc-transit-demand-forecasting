"""
Champion gate + ensemble weight analysis.

Both model families always go to Production (the ensemble uses both); the gate
only decides whether the newest version of each family replaces the previous
Production version of that family.

Promotion is based on the honest out-of-sample MAE that each training run logged
for its version. Production models are refit on the *full* dataset before being
registered, so re-forecasting them here would score them on data they trained on
(in-sample) and be meaningless — the training run is the single owner of the
holdout. Each training run also logs its per-day holdout predictions, which this
module reads to run the ensemble weight analysis on a common, out-of-sample window
for both families.

The recommended weight is reported, not auto-applied: tuning the shipped weight on
a short holdout each run would overfit and make the production weight unstable
month to month. After evaluation a training baseline JSON (ensemble MAE) is written
to S3 so the monitoring pipeline has a meaningful retrain threshold.
"""
import warnings

import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

from src.utils.config import (
    ENSEMBLE_SARIMAX_WEIGHT,
    ENSEMBLE_XGB_WEIGHT,
    GOLD_SARIMA_LOCAL_PATH,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    S3_TRAINING_BASELINE_KEY,
    SARIMAX_MODEL_NAME,
    TEST_DAYS,
    XGBOOST_MODEL_NAME,
)
from src.utils.logger import get_logger
from src.utils.s3_helpers import get_s3_client, write_s3_json

warnings.filterwarnings("ignore")
logger = get_logger(__name__)

HOLDOUT_ARTIFACT = "holdout_predictions.json"


def _latest_and_prod_versions(client: MlflowClient, model_name: str) -> tuple[int | None, int | None]:
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        return None, None
    new_ver = max(int(v.version) for v in versions)
    prod = [int(v.version) for v in versions if v.current_stage == "Production"]
    return new_ver, (max(prod) if prod else None)


def _logged_metric(client: MlflowClient, model_name: str, version: int, metric: str = "mae") -> float | None:
    v = client.get_model_version(model_name, str(version))
    return client.get_run(v.run_id).data.metrics.get(metric)


def _load_holdout(client: MlflowClient, model_name: str, version: int) -> pd.DataFrame | None:
    """Download the per-day holdout predictions logged by a training run."""
    v = client.get_model_version(model_name, str(version))
    try:
        path = mlflow.artifacts.download_artifacts(
            run_id=v.run_id, artifact_path=HOLDOUT_ARTIFACT, tracking_uri=MLFLOW_TRACKING_URI,
        )
    except Exception as e:
        logger.warning(f"No holdout artifact for {model_name} v{version}: {e}")
        return None
    return pd.read_json(path)


def _promote(client: MlflowClient, model_name: str, version: int) -> None:
    client.transition_model_version_stage(
        name=model_name, version=str(version), stage="Production",
        archive_existing_versions=True,
    )


def _gate(client: MlflowClient, model_name: str) -> tuple[int | None, float | None]:
    """Promote the newest version to Production iff its logged holdout MAE beats
    the current Production version's. Returns (candidate_version, candidate_mae)
    regardless of the promotion outcome, so the ensemble analysis can always use
    the freshly trained models.
    """
    new_ver, prod_ver = _latest_and_prod_versions(client, model_name)
    if new_ver is None:
        logger.info(f"{model_name} — no registered versions, skipping")
        return None, None

    new_mae = _logged_metric(client, model_name, new_ver)
    logger.info(f"{model_name} v{new_ver} (candidate) — holdout MAE: {new_mae}")

    if prod_ver is None:
        _promote(client, model_name, new_ver)
        logger.info(f"{model_name} v{new_ver} → Production (no prior champion)")
    elif prod_ver == new_ver:
        logger.info(f"{model_name} v{new_ver} already in Production — skipping gate")
    else:
        old_mae = _logged_metric(client, model_name, prod_ver)
        logger.info(f"{model_name} v{prod_ver} (Production) — holdout MAE: {old_mae}")
        if new_mae is not None and old_mae is not None and new_mae < old_mae:
            _promote(client, model_name, new_ver)
            logger.info(
                f"{model_name} v{new_ver} → Production "
                f"(MAE {new_mae:.4f}M < {old_mae:.4f}M, improvement: {old_mae - new_mae:.4f}M)"
            )
        else:
            logger.info(
                f"{model_name} v{new_ver} NOT promoted "
                f"(MAE {new_mae} >= {old_mae}) — keeping v{prod_ver} in Production"
            )

    return new_ver, new_mae


def evaluate_baselines() -> dict:
    """Naive benchmarks on the same holdout, for a 'compared to what?' reference.

    seasonal_naive_m7 forecasts each day as the value 7 days earlier (same
    weekday last week); persistence forecasts each day as the previous day.
    For daily ridership with strong weekly seasonality, seasonal-naive is the
    standard hard-to-beat benchmark — the models must beat it to justify their
    complexity. Informational only — never raises, so missing gold data cannot
    block champion selection or promotion.
    """
    try:
        df = pd.read_parquet(GOLD_SARIMA_LOCAL_PATH)
    except (FileNotFoundError, OSError) as e:
        logger.warning(f"Baselines skipped — gold SARIMA data unavailable: {e}")
        return {}
    df.index = pd.to_datetime(df.index)
    df = df.asfreq("D")
    y = df["daily_ridership"] / 1_000_000

    test_idx = y.iloc[-TEST_DAYS:].index
    y_test = y.loc[test_idx]

    results: dict = {}
    for name, shifted in (("seasonal_naive_m7", y.shift(7)), ("persistence", y.shift(1))):
        pred = shifted.loc[test_idx]
        mask = pred.notna() & y_test.notna()
        if int(mask.sum()) == 0:
            continue
        results[name] = {
            "mae": float(mean_absolute_error(y_test[mask], pred[mask])),
            "mape_pct": float(mean_absolute_percentage_error(y_test[mask], pred[mask]) * 100),
            "n_evaluated": int(mask.sum()),
        }
    return results


def grid_search_weight(
    sarimax_pred: np.ndarray, xgb_pred: np.ndarray, y_actual: np.ndarray, step: float = 0.05
) -> tuple[float, float, list[dict]]:
    """Sweep the SARIMAX ensemble weight over [0, 1] and return the MAE-minimizing
    weight, its MAE, and the full curve. Reported for ratification — not applied
    automatically, since tuning the shipped weight on a short holdout each run
    would overfit and make the production weight unstable month to month.
    """
    weights = np.round(np.arange(0.0, 1.0 + step / 2, step), 2)
    curve: list[dict] = []
    best_w, best_mae = ENSEMBLE_SARIMAX_WEIGHT, float("inf")
    for w in weights:
        ens = w * sarimax_pred + (1.0 - w) * xgb_pred
        mae = float(np.mean(np.abs(ens - y_actual)))
        curve.append({"sarimax_weight": float(w), "ensemble_mae": mae})
        if mae < best_mae:
            best_mae, best_w = mae, float(w)
    return best_w, best_mae, curve


def run() -> str:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    logger.info("Gating SARIMAX")
    sar_ver, sar_mae = _gate(client, SARIMAX_MODEL_NAME)
    logger.info("Gating XGBoost")
    xgb_ver, xgb_mae = _gate(client, XGBOOST_MODEL_NAME)

    if sar_mae is None or xgb_mae is None:
        logger.warning("Missing logged holdout MAE for a family — skipping ensemble analysis")
        return SARIMAX_MODEL_NAME if (sar_mae or float("inf")) <= (xgb_mae or float("inf")) else XGBOOST_MODEL_NAME

    winner = SARIMAX_MODEL_NAME if sar_mae <= xgb_mae else XGBOOST_MODEL_NAME
    logger.info(f"Champion model family: {winner} (MAE {min(sar_mae, xgb_mae):.4f}M)")

    # ── Ensemble weight analysis on the common, out-of-sample holdout ──────────
    sar_h = _load_holdout(client, SARIMAX_MODEL_NAME, sar_ver)
    xgb_h = _load_holdout(client, XGBOOST_MODEL_NAME, xgb_ver)
    if sar_h is None or xgb_h is None:
        logger.warning("Missing holdout predictions — skipping ensemble weight analysis")
        return winner

    merged = sar_h.merge(xgb_h, on="date", suffixes=("_sar", "_xgb"))
    if merged.empty:
        logger.warning("SARIMAX/XGBoost holdout windows do not overlap — skipping ensemble analysis")
        return winner

    y_actual = merged["y_true_sar"].to_numpy()
    sarimax_pred = merged["y_pred_sar"].to_numpy()
    xgb_pred = merged["y_pred_xgb"].to_numpy()

    # Ensemble at the shipped (config) weight — this is what production uses.
    ensemble_pred = ENSEMBLE_SARIMAX_WEIGHT * sarimax_pred + ENSEMBLE_XGB_WEIGHT * xgb_pred
    ensemble_mae = float(np.mean(np.abs(ensemble_pred - y_actual)))

    # Data-driven weight recommendation (reported, not auto-applied).
    best_w, best_w_mae, weight_curve = grid_search_weight(sarimax_pred, xgb_pred, y_actual)

    # Naive baselines for context.
    baselines = evaluate_baselines()
    sn_mae = baselines.get("seasonal_naive_m7", {}).get("mae", float("nan"))
    pers_mae = baselines.get("persistence", {}).get("mae", float("nan"))

    logger.info(
        "Holdout MAE (M) — "
        f"seasonal_naive: {sn_mae:.4f} | persistence: {pers_mae:.4f} | "
        f"sarimax: {sar_mae:.4f} | xgboost: {xgb_mae:.4f} | "
        f"ensemble@{ENSEMBLE_SARIMAX_WEIGHT:.2f}: {ensemble_mae:.4f} | "
        f"ensemble@best({best_w:.2f}): {best_w_mae:.4f}"
    )
    if ensemble_mae >= sn_mae:
        logger.warning(
            f"Ensemble (MAE {ensemble_mae:.4f}M) does NOT beat seasonal-naive "
            f"(MAE {sn_mae:.4f}M) on this holdout — investigate before trusting forecasts."
        )
    if abs(best_w - ENSEMBLE_SARIMAX_WEIGHT) > 1e-9 and (ensemble_mae - best_w_mae) > 0.001:
        logger.warning(
            f"Config SARIMAX weight {ENSEMBLE_SARIMAX_WEIGHT:.2f} is sub-optimal on this "
            f"holdout; best={best_w:.2f} (MAE {best_w_mae:.4f}M vs {ensemble_mae:.4f}M). "
            f"Consider updating ENSEMBLE_SARIMAX_WEIGHT in config if this persists."
        )

    s3 = get_s3_client()
    write_s3_json(s3, {
        "ensemble_mae": ensemble_mae,
        "sarimax_mae": sar_mae,
        "xgboost_mae": xgb_mae,
        "champion_model": winner,
        "config_sarimax_weight": ENSEMBLE_SARIMAX_WEIGHT,
        "recommended_sarimax_weight": best_w,
        "recommended_weight_mae": best_w_mae,
        "weight_curve": weight_curve,
        "baselines": baselines,
        "n_holdout": int(len(y_actual)),
    }, S3_TRAINING_BASELINE_KEY)
    logger.info("Training baseline + weight analysis written to S3")

    return winner


if __name__ == "__main__":
    run()
