"""
Champion gate + ensemble weight analysis.

Both model families always go to Production (the ensemble uses both); the gate
only decides whether the newest version of each family replaces the previous
Production version of that family.

Promotion is head-to-head on one window: the candidate's TEST_DAYS holdout. The
candidate side is the per-day holdout forecast its training run logged (made by an
eval model fit only on data before the window — the registered candidate is a
full-data refit, so it can't be scored there). The Production side is that version
re-scored here on the same days, through the serving code path: SARIMAX re-anchored
on actuals up to the day before the window with its own scaler, XGBoost forecast
iteratively from lags known at that point. The better of the two is promoted; a
seasonal-naive forecast made from the same origin is logged alongside, with a
warning when the served version doesn't beat it.

Comparing each version's own logged MAE instead (the earlier design) compared
different windows: a version scored on an easy early-summer month kept beating every
later candidate scored on a harder one, and Production went four months without a
promotion. When Production can't be scored fairly — its training data reaches into
the window (e.g. a drift retrain soon after the last one), or its artifacts or
feature set no longer load — the gate falls back to the naive floor alone, so it can
never freeze again. Every decision is recorded in the training baseline JSON.

Each training run also logs its per-day holdout predictions, which this module reads
to run the ensemble weight analysis on a common, out-of-sample window for both
families.

The recommended weight is reported, not auto-applied: tuning the shipped weight on
a short holdout each run would overfit and make the production weight unstable
month to month. After evaluation a training baseline JSON (ensemble MAE) is written
to S3 as the monitoring pipeline's fallback retrain threshold — monitoring prefers
the walk-forward ensemble MAE, which matches the live metric's horizon and cadence.
The baseline's ensemble MAE is computed from the freshly trained candidates even
when one failed its gate, so the JSON records both the candidate versions it was
computed from and the post-gate Production versions, making any mismatch visible.
"""
import math
import pickle
import warnings

import mlflow
import mlflow.statsmodels
import mlflow.xgboost
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

from src.utils.config import (
    ENSEMBLE_SARIMAX_WEIGHT,
    ENSEMBLE_XGB_WEIGHT,
    GOLD_ML_LOCAL_PATH,
    GOLD_SARIMA_LOCAL_PATH,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    S3_TRAINING_BASELINE_KEY,
    SARIMAX_EXOG_COLS,
    SARIMAX_MODEL_NAME,
    TEST_DAYS,
    XGBOOST_MODEL_NAME,
)
from src.utils.features import cast_categoricals, iterative_xgb_predict
from src.utils.logger import get_logger
from src.utils.registry import promote_to_production, resolve_production_version
from src.utils.s3_helpers import get_s3_client, write_s3_json

warnings.filterwarnings("ignore")
logger = get_logger(__name__)

HOLDOUT_ARTIFACT = "holdout_predictions.json"


def _latest_and_prod_versions(client: MlflowClient, model_name: str) -> tuple[int | None, int | None]:
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        return None, None
    new_ver = max(int(v.version) for v in versions)
    prod = resolve_production_version(client, model_name)
    return new_ver, (int(prod.version) if prod else None)


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
    promote_to_production(client, model_name, version)


def _load_gold_sarima() -> pd.DataFrame | None:
    try:
        df = pd.read_parquet(GOLD_SARIMA_LOCAL_PATH)
    except (FileNotFoundError, OSError) as e:
        logger.warning(f"Gold SARIMA data unavailable: {e}")
        return None
    df.index = pd.to_datetime(df.index)
    return df.asfreq("D")


def seasonal_naive_from_origin(y: pd.Series, dates: pd.DatetimeIndex) -> np.ndarray:
    """Seasonal-naive forecast of `dates` made from the day before the first one:
    each day copies the same weekday from the latest week known at that origin
    (last week for days 1-7, two weeks back for days 8-14, ...). Unlike y.shift(7),
    it never reads actuals inside the window, so it faces the same multi-step task
    as the models it is a floor for."""
    origin = dates[0] - pd.Timedelta(days=1)
    return np.array([
        y.loc[d - pd.Timedelta(days=7 * math.ceil((d - origin).days / 7))] for d in dates
    ])


def promotion_decision(
    candidate_mae: float, production_mae: float | None, naive_mae: float
) -> tuple[bool, str]:
    """Returns (promote, method). When Production was scored on the same window, the
    better of the two servable models wins: naive isn't servable, so requiring the
    candidate to beat it too would keep an even worse Production model. Only when
    Production can't be scored does the naive floor decide, so a broken candidate
    still can't ship."""
    if production_mae is None:
        return candidate_mae < naive_mae, "naive_floor"
    return candidate_mae < production_mae, "head_to_head"


def _score_sarimax(client: MlflowClient, version: int, dates: pd.DatetimeIndex) -> np.ndarray | None:
    # Reuses the serving re-anchor so Production is scored exactly as it forecasts.
    # Imported here so the evaluation module doesn't import prediction at load time.
    from src.prediction.generate_forecast import _reanchor_sarimax

    model = mlflow.statsmodels.load_model(f"models:/{SARIMAX_MODEL_NAME}/{version}")
    if model.fittedvalues.index[-1] >= dates[0]:
        logger.info(f"{SARIMAX_MODEL_NAME} v{version} trained into the window — not scorable")
        return None
    run_id = client.get_model_version(SARIMAX_MODEL_NAME, str(version)).run_id
    scaler_path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path="sarimax_scaler.pkl", tracking_uri=MLFLOW_TRACKING_URI,
    )
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    gold = _load_gold_sarima()
    origin = dates[0] - pd.Timedelta(days=1)
    model = _reanchor_sarimax(model, gold.loc[:origin], scaler)
    exog = pd.DataFrame(
        scaler.transform(gold.loc[dates, SARIMAX_EXOG_COLS]), index=dates, columns=SARIMAX_EXOG_COLS
    )
    return np.asarray(model.get_forecast(steps=len(dates), exog=exog).predicted_mean)


def _score_xgboost(client: MlflowClient, version: int, dates: pd.DatetimeIndex) -> np.ndarray | None:
    run_id = client.get_model_version(XGBOOST_MODEL_NAME, str(version)).run_id
    trained_through = client.get_run(run_id).data.tags.get("data_version_date")
    if trained_through is None or pd.Timestamp(trained_through) >= dates[0]:
        logger.info(f"{XGBOOST_MODEL_NAME} v{version} trained into the window (or unknown) — not scorable")
        return None
    model = mlflow.xgboost.load_model(f"models:/{XGBOOST_MODEL_NAME}/{version}")
    df_ml = pd.read_parquet(GOLD_ML_LOCAL_PATH)
    df_ml.index = pd.to_datetime(df_ml.index)
    df_ml = cast_categoricals(df_ml)
    return iterative_xgb_predict(model, df_ml, df_ml.index.get_loc(dates[0]), len(dates))


_PRODUCTION_SCORERS = {SARIMAX_MODEL_NAME: _score_sarimax, XGBOOST_MODEL_NAME: _score_xgboost}


def _score_production(
    client: MlflowClient, model_name: str, version: int, dates: pd.DatetimeIndex
) -> np.ndarray | None:
    """Production version's forecast of the candidate's holdout days, or None when
    it can't be scored fairly (trained into the window) or at all (missing
    artifact, feature set changed since it was trained, ...)."""
    try:
        pred = _PRODUCTION_SCORERS[model_name](client, version, dates)
    except Exception as e:
        logger.warning(f"Could not score {model_name} v{version} on the holdout window: {e}")
        return None
    if pred is not None and len(pred) != len(dates):
        logger.warning(f"{model_name} v{version} returned {len(pred)} predictions for {len(dates)} days")
        return None
    return pred


def _gate(client: MlflowClient, model_name: str) -> tuple[int | None, float | None, dict]:
    """Promote the newest version if it beats Production head-to-head on its holdout
    window (or, if Production can't be scored, seasonal-naive). Returns
    (candidate_version, candidate_mae, record)
    regardless of the outcome, so the ensemble analysis can always use the freshly
    trained models; the record explains the decision.
    """
    new_ver, prod_ver = _latest_and_prod_versions(client, model_name)
    if new_ver is None:
        logger.info(f"{model_name} — no registered versions, skipping")
        return None, None, {}

    new_mae = _logged_metric(client, model_name, new_ver)
    logger.info(f"{model_name} v{new_ver} (candidate) — holdout MAE: {new_mae}")
    record: dict = {"candidate_version": new_ver, "candidate_mae": new_mae, "production_version": prod_ver}

    if prod_ver is None:
        _promote(client, model_name, new_ver)
        logger.info(f"{model_name} v{new_ver} → Production (no prior champion)")
        return new_ver, new_mae, {**record, "method": "first_promotion", "promoted": True}
    if prod_ver == new_ver:
        logger.info(f"{model_name} v{new_ver} already in Production — skipping gate")
        return new_ver, new_mae, {**record, "method": "already_production", "promoted": False}

    holdout = _load_holdout(client, model_name, new_ver)
    gold = _load_gold_sarima()
    if holdout is None or gold is None or new_mae is None:
        logger.warning(f"{model_name} v{new_ver} NOT promoted — no holdout or gold data to compare on")
        return new_ver, new_mae, {**record, "method": "no_comparison", "promoted": False}

    dates = pd.DatetimeIndex(pd.to_datetime(holdout["date"]))
    y_true = holdout["y_true"].to_numpy()
    y = gold["daily_ridership"] / 1_000_000
    naive_mae = float(np.mean(np.abs(seasonal_naive_from_origin(y, dates) - y_true)))
    prod_pred = _score_production(client, model_name, prod_ver, dates)
    prod_mae = float(np.mean(np.abs(prod_pred - y_true))) if prod_pred is not None else None

    promote, method = promotion_decision(new_mae, prod_mae, naive_mae)
    window = [str(dates[0].date()), str(dates[-1].date())]
    record.update({"method": method, "promoted": promote, "production_mae": prod_mae,
                   "naive_mae": naive_mae, "window": window})
    prod_txt = (f"Production v{prod_ver} {prod_mae:.4f}M" if prod_mae is not None
                else f"Production v{prod_ver} not scorable")
    summary = (f"{window[0]}..{window[1]}: candidate {new_mae:.4f}M, {prod_txt}, "
               f"seasonal-naive {naive_mae:.4f}M")
    if promote:
        _promote(client, model_name, new_ver)
        logger.info(f"{model_name} v{new_ver} → Production ({method}; {summary})")
    else:
        logger.info(f"{model_name} v{new_ver} NOT promoted ({method}; {summary}) — keeping v{prod_ver}")
    served_mae = new_mae if promote else prod_mae
    if served_mae is not None and served_mae >= naive_mae:
        logger.warning(f"{model_name}: the served version does not beat seasonal-naive ({summary})")
    return new_ver, new_mae, record


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
    sar_ver, sar_mae, sar_gate = _gate(client, SARIMAX_MODEL_NAME)
    logger.info("Gating XGBoost")
    xgb_ver, xgb_mae, xgb_gate = _gate(client, XGBOOST_MODEL_NAME)

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

    # The ensemble analysis above deliberately uses the freshly trained candidates
    # (freshest common holdout). When a candidate failed its gate, Production serves
    # an older version and ensemble_mae describes a pair that isn't live — record
    # both version sets so the mismatch is visible in the file, never silent.
    prod_sar = resolve_production_version(client, SARIMAX_MODEL_NAME)
    prod_xgb = resolve_production_version(client, XGBOOST_MODEL_NAME)
    baseline_versions = {"sarimax": sar_ver, "xgboost": xgb_ver}
    production_versions = {
        "sarimax": int(prod_sar.version) if prod_sar else None,
        "xgboost": int(prod_xgb.version) if prod_xgb else None,
    }
    if baseline_versions != production_versions:
        logger.warning(
            f"Baseline ensemble MAE computed from candidate versions {baseline_versions} "
            f"but Production serves {production_versions} — the baseline describes a "
            f"model pair that is not live"
        )

    s3 = get_s3_client()
    write_s3_json(s3, {
        "ensemble_mae": ensemble_mae,
        "sarimax_mae": sar_mae,
        "xgboost_mae": xgb_mae,
        "baseline_versions": baseline_versions,
        "production_versions": production_versions,
        "gate": {"sarimax": sar_gate, "xgboost": xgb_gate},
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
