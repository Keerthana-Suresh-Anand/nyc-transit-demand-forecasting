"""
Champion gate + ensemble weight analysis.

Both model families always go to Production (the ensemble uses both); the gate
only decides whether the newest version of each family replaces the previous
Production version of that family.

Evidence. Each training run logs its candidate's holdout as weekly 14-day forecasts
over the last TEST_DAYS (``HOLDOUT_FORECASTS_ARTIFACT``), each made from the actuals
known at its origin by an eval model fit only on data before the window — the task
production serves. The registered candidate is a full-data refit, so this logged
holdout is the only out-of-sample evidence for it.

Head-to-head. The Production version is re-scored here from the same origins
through the serving path (SARIMAX re-anchored on actuals up to the origin with its
own scaler; XGBoost iterative from lags known at the origin), but only from origins
on or after the end of its own training data — earlier origins would be in-sample
for it. Candidate and Production are compared on exactly those origins and the
better one serves. A same-origin seasonal-naive forecast is logged alongside, with
a warning when the served version doesn't beat it.

Fallback. If no origin postdates Production's training data (e.g. a drift retrain
days after the last one), or Production can't be scored at all, the candidate must
beat the same-origin seasonal-naive on all origins. Every decision is recorded in
the training baseline JSON.

History. The first gate compared each version's own logged MAE — different windows,
so a version scored on an easy month blocked every later candidate for four months.
The second compared both on the candidate's 30-day window as one 30-step forecast,
which Production could almost never be scored on (its training data overlapped
the window whenever retrains were < 30 days apart) and which compounded XGBoost's
recursive error over horizons production never serves.

The ensemble weight analysis runs on the two candidates' logged holdout forecasts.
The recommended weight is reported, not auto-applied: tuning the shipped weight on
a short holdout each run would overfit and make the production weight unstable
month to month. A training baseline JSON (ensemble MAE) is written to S3 as the
monitoring pipeline's fallback retrain threshold — monitoring prefers the
walk-forward ensemble MAE. That baseline is computed from the freshly trained
candidates even when one failed its gate, so the JSON records both the candidate
versions and the post-gate Production versions, making any mismatch visible.
"""
import pickle
import warnings

import mlflow
import mlflow.statsmodels
import mlflow.xgboost
import numpy as np
import pandas as pd
from mlflow import MlflowClient

from src.evaluation.origins import persistence_path, seasonal_naive_path
from src.utils.config import (
    ENSEMBLE_SARIMAX_WEIGHT,
    ENSEMBLE_XGB_WEIGHT,
    FORECAST_HORIZON_DAYS,
    GOLD_ML_LOCAL_PATH,
    GOLD_SARIMA_LOCAL_PATH,
    HOLDOUT_FORECASTS_ARTIFACT,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    S3_TRAINING_BASELINE_KEY,
    SARIMAX_EXOG_COLS,
    SARIMAX_MODEL_NAME,
    XGBOOST_MODEL_NAME,
)
from src.utils.features import cast_categoricals, iterative_xgb_predict
from src.utils.logger import get_logger
from src.utils.registry import promote_to_production, resolve_production_version
from src.utils.s3_helpers import get_s3_client, write_s3_json

warnings.filterwarnings("ignore")
logger = get_logger(__name__)

H = FORECAST_HORIZON_DAYS


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
    """Per-origin holdout forecasts logged by a training run (origin, date, step,
    y_true, y_pred), or None for versions trained before they were logged."""
    v = client.get_model_version(model_name, str(version))
    try:
        path = mlflow.artifacts.download_artifacts(
            run_id=v.run_id, artifact_path=HOLDOUT_FORECASTS_ARTIFACT,
            tracking_uri=MLFLOW_TRACKING_URI,
        )
    except Exception as e:
        logger.warning(f"No holdout forecasts for {model_name} v{version}: {e}")
        return None
    df = pd.read_json(path)
    df["origin"] = pd.to_datetime(df["origin"])
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["origin", "step"]).reset_index(drop=True)


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


def _mae(pred: np.ndarray, actual: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(pred) - np.asarray(actual))))


def baseline_predictions(y: pd.Series, records: pd.DataFrame, kind: str) -> np.ndarray:
    """Same-origin naive forecasts aligned with `records` (sorted by origin, step):
    for each origin, a path built only from actuals known at that origin."""
    path_fn = {"seasonal_naive": seasonal_naive_path, "persistence": persistence_path}[kind]
    return np.concatenate([
        path_fn(y.loc[:origin].to_numpy(), len(grp))
        for origin, grp in records.groupby("origin", sort=True)
    ])


def promotion_decision(
    candidate_mae: float, production_mae: float | None, naive_mae: float
) -> tuple[bool, str]:
    """Returns (promote, method). When Production was scored on the same origins,
    the better of the two servable models wins: naive isn't servable, so requiring
    the candidate to beat it too would keep an even worse Production model. Only when
    Production can't be scored does the naive floor decide, so a broken candidate
    still can't ship."""
    if production_mae is None:
        return candidate_mae < naive_mae, "naive_floor"
    return candidate_mae < production_mae, "head_to_head"


def _score_sarimax(
    client: MlflowClient, version: int, origins: list[pd.Timestamp]
) -> tuple[pd.Timestamp, dict[pd.Timestamp, np.ndarray]]:
    """(training cutoff, {origin: 14-day forecast}) for origins on/after the cutoff."""
    # Reuses the serving re-anchor so Production is scored exactly as it forecasts.
    # Imported here so the evaluation module doesn't import prediction at load time.
    from src.prediction.generate_forecast import _reanchor_sarimax

    model = mlflow.statsmodels.load_model(f"models:/{SARIMAX_MODEL_NAME}/{version}")
    cutoff = model.fittedvalues.index[-1]
    eligible = [o for o in origins if o >= cutoff]
    if not eligible:
        return cutoff, {}
    run_id = client.get_model_version(SARIMAX_MODEL_NAME, str(version)).run_id
    scaler_path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path="sarimax_scaler.pkl", tracking_uri=MLFLOW_TRACKING_URI,
    )
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    gold = _load_gold_sarima()
    exog = pd.DataFrame(scaler.transform(gold[SARIMAX_EXOG_COLS]), index=gold.index,
                        columns=SARIMAX_EXOG_COLS)
    out = {}
    for origin in eligible:
        anchored = _reanchor_sarimax(model, gold.loc[:origin], scaler)
        dates = pd.date_range(origin + pd.Timedelta(days=1), periods=H, freq="D")
        out[origin] = np.asarray(
            anchored.get_forecast(steps=H, exog=exog.loc[dates]).predicted_mean)
    return cutoff, out


def _score_xgboost(
    client: MlflowClient, version: int, origins: list[pd.Timestamp]
) -> tuple[pd.Timestamp | None, dict[pd.Timestamp, np.ndarray]]:
    """(training cutoff, {origin: 14-day forecast}) for origins on/after the cutoff.
    An unknown cutoff means nothing can be scored fairly."""
    run_id = client.get_model_version(XGBOOST_MODEL_NAME, str(version)).run_id
    trained_through = client.get_run(run_id).data.tags.get("data_version_date")
    if trained_through is None:
        return None, {}
    cutoff = pd.Timestamp(trained_through)
    eligible = [o for o in origins if o >= cutoff]
    if not eligible:
        return cutoff, {}
    model = mlflow.xgboost.load_model(f"models:/{XGBOOST_MODEL_NAME}/{version}")
    df_ml = pd.read_parquet(GOLD_ML_LOCAL_PATH)
    df_ml.index = pd.to_datetime(df_ml.index)
    df_ml = cast_categoricals(df_ml)
    return cutoff, {
        o: iterative_xgb_predict(model, df_ml, df_ml.index.get_loc(o) + 1, H) for o in eligible
    }


_PRODUCTION_SCORERS = {SARIMAX_MODEL_NAME: _score_sarimax, XGBOOST_MODEL_NAME: _score_xgboost}


def _score_production(
    client: MlflowClient, model_name: str, version: int, origins: list[pd.Timestamp]
) -> tuple[pd.Timestamp | None, dict[pd.Timestamp, np.ndarray]]:
    """Production version's forecasts from the holdout origins it can be scored on
    fairly (on/after its training cutoff). Returns (cutoff, {origin: forecast}); the
    dict is empty when no origin qualifies or scoring fails (missing artifact,
    feature set changed since it was trained, ...)."""
    try:
        cutoff, preds = _PRODUCTION_SCORERS[model_name](client, version, origins)
    except Exception as e:
        logger.warning(f"Could not score {model_name} v{version} on the holdout origins: {e}")
        return None, {}
    bad = [o for o, p in preds.items() if len(p) != H]
    if bad:
        logger.warning(f"{model_name} v{version} returned short forecasts for {bad}")
        return cutoff, {}
    return cutoff, preds


def _gate(client: MlflowClient, model_name: str) -> tuple[int | None, float | None, dict]:
    """Promote the newest version if it beats Production head-to-head on the holdout
    origins Production can be scored on (or, if there are none, seasonal-naive).
    Returns (candidate_version, candidate_mae, record) regardless of the outcome, so
    the ensemble analysis can always use the freshly trained models; the record
    explains the decision.
    """
    new_ver, prod_ver = _latest_and_prod_versions(client, model_name)
    if new_ver is None:
        logger.info(f"{model_name} — no registered versions, skipping")
        return None, None, {}

    new_mae = _logged_metric(client, model_name, new_ver)
    logger.info(f"{model_name} v{new_ver} (candidate) — holdout MAE: {new_mae}")
    record: dict = {"candidate_version": new_ver, "candidate_holdout_mae": new_mae,
                    "production_version": prod_ver}

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

    y = gold["daily_ridership"] / 1_000_000
    origins = [pd.Timestamp(o) for o in sorted(holdout["origin"].unique())]
    cutoff, prod_preds = _score_production(client, model_name, prod_ver, origins)
    compared = sorted(prod_preds) if prod_preds else origins
    rows = holdout[holdout["origin"].isin(compared)]

    candidate_mae = _mae(rows["y_pred"], rows["y_true"])
    naive_mae = _mae(baseline_predictions(y, rows, "seasonal_naive"), rows["y_true"])
    prod_mae = (
        _mae(np.concatenate([prod_preds[o] for o in compared]), rows["y_true"])
        if prod_preds else None
    )

    promote, method = promotion_decision(candidate_mae, prod_mae, naive_mae)
    record.update({
        "method": method, "promoted": promote,
        "production_cutoff": str(cutoff.date()) if cutoff is not None else None,
        "origins": [str(pd.Timestamp(o).date()) for o in compared],
        "n_points": int(len(rows)),
        "candidate_mae": candidate_mae, "production_mae": prod_mae, "naive_mae": naive_mae,
    })
    prod_txt = (f"Production v{prod_ver} {prod_mae:.4f}M" if prod_mae is not None
                else f"Production v{prod_ver} not scorable (cutoff {record['production_cutoff']})")
    summary = (f"{len(compared)} origin(s) {record['origins'][0]}..{record['origins'][-1]}: "
               f"candidate {candidate_mae:.4f}M, {prod_txt}, seasonal-naive {naive_mae:.4f}M")
    if promote:
        _promote(client, model_name, new_ver)
        logger.info(f"{model_name} v{new_ver} → Production ({method}; {summary})")
    else:
        logger.info(f"{model_name} v{new_ver} NOT promoted ({method}; {summary}) — keeping v{prod_ver}")
    served_mae = candidate_mae if promote else prod_mae
    if served_mae is not None and served_mae >= naive_mae:
        logger.warning(f"{model_name}: the served version does not beat seasonal-naive ({summary})")
    return new_ver, new_mae, record


def evaluate_baselines(y: pd.Series, records: pd.DataFrame) -> dict:
    """Same-origin naive benchmarks on the holdout rows, the 'compared to what?'.

    seasonal_naive_m7 copies the same weekday from the latest week known at each
    origin; persistence holds the origin's value flat. Both face the same 14-day
    multi-step task as the models and match the walk-forward's definitions, so the
    holdout and backtest numbers are comparable. Informational only.
    """
    out: dict = {}
    for key, kind in (("seasonal_naive_m7", "seasonal_naive"), ("persistence", "persistence")):
        pred = baseline_predictions(y, records, kind)
        actual = records["y_true"].to_numpy()
        out[key] = {
            "mae": _mae(pred, actual),
            "mape_pct": float(np.mean(np.abs(pred - actual) / actual) * 100),
            "n_evaluated": int(len(actual)),
        }
    return out


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
        logger.warning("Missing holdout forecasts — skipping ensemble weight analysis")
        return winner

    merged = sar_h.merge(xgb_h[["origin", "date", "y_pred"]], on=["origin", "date"],
                         suffixes=("_sar", "_xgb")).sort_values(["origin", "step"])
    if merged.empty:
        logger.warning("SARIMAX/XGBoost holdout origins do not overlap — skipping ensemble analysis")
        return winner

    y_actual = merged["y_true"].to_numpy()
    sarimax_pred = merged["y_pred_sar"].to_numpy()
    xgb_pred = merged["y_pred_xgb"].to_numpy()

    # Ensemble at the shipped (config) weight — this is what production uses.
    ensemble_pred = ENSEMBLE_SARIMAX_WEIGHT * sarimax_pred + ENSEMBLE_XGB_WEIGHT * xgb_pred
    ensemble_mae = _mae(ensemble_pred, y_actual)

    # Data-driven weight recommendation (reported, not auto-applied).
    best_w, best_w_mae, weight_curve = grid_search_weight(sarimax_pred, xgb_pred, y_actual)

    gold = _load_gold_sarima()
    baselines = (evaluate_baselines(gold["daily_ridership"] / 1_000_000, merged)
                 if gold is not None else {})
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
        logger.info(
            f"Holdout-optimal SARIMAX weight {best_w:.2f} (MAE {best_w_mae:.4f}M vs "
            f"{ensemble_mae:.4f}M at {ENSEMBLE_SARIMAX_WEIGHT:.2f}) — reported only; the "
            "shipped weight follows the walk-forward + significance test, not one holdout."
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
        "n_holdout_origins": int(merged["origin"].nunique()),
    }, S3_TRAINING_BASELINE_KEY)
    logger.info("Training baseline + weight analysis written to S3")

    return winner


if __name__ == "__main__":
    run()
