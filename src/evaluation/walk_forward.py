"""14-day rolling-origin walk-forward evaluation + block-bootstrap significance.

This is the evidence behind the 50/50 ensemble weight and the "keep both models"
decision (see CLAUDE.md → "Model evaluation methodology"). It evaluates SARIMAX,
XGBoost, naive baselines, and the ensemble the way production actually serves
them: a 14-day forecast re-anchored weekly. Models are trained once on data up to
the eval window; at each weekly origin SARIMAX state is updated with observed
actuals (`.append`, params fixed) and XGBoost forecasts 14 days iteratively,
seeding lags from actuals known at that origin. This removes the unfair 60-day
compounding that a single long holdout imposes on XGBoost.

Runs at the end of every training pipeline (best-effort — a failure never undoes
the registration that already succeeded), which writes the result to S3 as the
dashboard's headline accuracy, the monitoring baseline and the conformal band's
calibration. Can also be run on demand; `run()` itself only reads S3 (the pinned
SARIMAX order) and never touches MLflow. The statistical core
(`block_bootstrap_mae_diff`, `summarize`) is pure and unit-tested; `run()` does the
heavy model fitting.
"""
import math
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.evaluation.evaluate_models import grid_search_weight
from src.evaluation.origins import (
    ORIGIN_STEP,
    forecast_origins,
    persistence_path,
    sarimax_origin_forecasts,
    seasonal_naive_path,
    xgb_origin_forecasts,
)
from src.training.train_sarimax import EXOG_COLS, _load_cached_order, find_best_params
from src.training.train_xgboost import get_feature_cols
from src.training.tune_xgboost import LIBRARY_DEFAULTS, load_serving_params
from src.training.xgb_recipe import fit_recipe
from src.transformation import preprocess_ml
from src.utils.config import (
    CI_COVERAGE,
    ENSEMBLE_SARIMAX_WEIGHT,
    ENSEMBLE_XGB_WEIGHT,
    FORECAST_HORIZON_DAYS,
    GOLD_ML_LOCAL_PATH,
    GOLD_SARIMA_LOCAL_PATH,
)
from src.utils.features import cast_categoricals
from src.utils.logger import get_logger
from src.utils.s3_helpers import get_s3_client

warnings.filterwarnings("ignore")
warnings.filterwarnings("always", category=ConvergenceWarning)  # keep convergence signal visible
logger = get_logger(__name__)

EVAL_DAYS = 90   # window over which origins roll
STEP = ORIGIN_STEP  # weekly re-anchor, matching production cadence
H = FORECAST_HORIZON_DAYS


def _mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(mean_absolute_error(a, b))


def _mape(a: np.ndarray, b: np.ndarray) -> float:
    return float(mean_absolute_percentage_error(a, b) * 100)


def block_bootstrap_mae_diff(p1_blocks, p2_blocks, actual_blocks, n_boot=10000, seed=0):
    """Block bootstrap of MAE(p1) − MAE(p2).

    Resamples whole forecast origins with replacement (not individual days) so the
    within-window autocorrelation is respected. Returns (mean_diff, lo95, hi95):
      diff < 0 across the whole CI  -> p1 reliably better
      diff > 0 across the whole CI  -> p2 reliably better
      CI spans 0                    -> difference is noise (tie)
    """
    rng = np.random.default_rng(seed)
    k = len(actual_blocks)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, k, k)
        a = np.concatenate([actual_blocks[i] for i in idx])
        d1 = np.mean(np.abs(np.concatenate([p1_blocks[i] for i in idx]) - a))
        d2 = np.mean(np.abs(np.concatenate([p2_blocks[i] for i in idx]) - a))
        diffs[b] = d1 - d2
    return float(diffs.mean()), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def _lead_time_scale(per_step: np.ndarray) -> np.ndarray:
    """Least-squares line through per-step error levels, with the slope floored at
    zero — error shouldn't shrink with lead time, and a negative slope is noise."""
    steps = np.arange(1, len(per_step) + 1)
    if len(per_step) < 2:
        return per_step.astype(float)
    mean = float(np.mean(per_step))
    slope, intercept = np.polyfit(steps, per_step, 1)
    if slope < 0:
        return np.full(len(per_step), mean)
    # A steep line can cross zero at short lead times; keep every step positive.
    return np.maximum(intercept + slope * steps, 0.1 * mean)


def conformal_half_widths(
    pred_blocks: list[np.ndarray], actual_blocks: list[np.ndarray], coverage: float,
) -> dict:
    """Split-conformal band half-widths for a forecast, one per horizon step.

    Scaled conformal: each |residual| is normalized by a lead-time scale, the
    conformal quantile is taken over the pooled scores (all origins × steps), and
    each step's half-width is quantile × that step's scale. Pooling makes the
    quantile estimable (n = origins × horizon rather than just origins per step).
    Coverage is therefore marginal across the horizon, not per-step — and the usual
    time-series caveat applies: residuals from rolled origins are not strictly
    exchangeable, so the guarantee is approximate.

    The scale is a non-decreasing linear trend fitted to each step's median
    |residual|, not each step's raw mean. Backtest origins are a week apart, so
    step k always lands on the same weekday: a raw per-step mean measures that
    weekday's error as much as lead time, and with ~11 residuals per step one bad
    day can dominate it (a single partial-data day once made two steps ~4x wider
    than the rest). The trend keeps only the lead-time signal.
    """
    pred = np.vstack(pred_blocks)            # (n_origins, horizon)
    actual = np.vstack(actual_blocks)
    abs_resid = np.abs(pred - actual)
    scale = np.maximum(_lead_time_scale(np.median(abs_resid, axis=0)), 1e-9)
    scores = (abs_resid / scale).ravel()
    n = scores.size
    k = min(math.ceil((n + 1) * coverage), n)
    q = float(np.sort(scores)[k - 1])
    return {
        "coverage": float(coverage),
        "quantile": q,
        "n_scores": int(n),
        "scale_by_horizon": [float(s) for s in scale],
        "half_width_by_horizon": [float(q * s) for s in scale],
    }


def significance_verdict(lo: float, hi: float, p1: str, p2: str) -> str:
    if hi < 0:
        return f"{p1} significantly better"
    if lo > 0:
        return f"{p2} significantly better"
    return "TIE (95% CI includes 0)"


def walk_forward(
    y: pd.Series, exog: pd.DataFrame, df_ml: pd.DataFrame,
    *, eval_days: int = EVAL_DAYS, step: int = STEP, horizon: int = H, order=None,
    xgb_params: dict | None = None,
) -> dict:
    """Fit the two models once, then roll a `horizon`-day origin forward weekly over
    the last `eval_days`. Returns per-origin prediction blocks (lists of arrays) for
    each model + the actuals and naive baselines.
    """
    n = len(y)
    train_end = n - eval_days  # positions [0:train_end] are the initial training set

    scaler = MinMaxScaler().fit(exog.iloc[:train_end])
    train_exog_s = pd.DataFrame(scaler.transform(exog.iloc[:train_end]),
                                index=y.index[:train_end], columns=EXOG_COLS)
    if order is None:
        order_pdq, seasonal = find_best_params(y.iloc[:train_end], train_exog_s)
    else:
        order_pdq, seasonal = order
    logger.info(f"Base SARIMAX{order_pdq}x{seasonal}")
    base = SARIMAX(y.iloc[:train_end], exog=train_exog_s, order=order_pdq, seasonal_order=seasonal,
                   enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)

    feat = get_feature_cols(df_ml)
    cutoff_date = y.index[train_end - 1]
    ml_cut = df_ml.index.get_loc(cutoff_date) + 1
    x_ml = df_ml[feat]
    y_ml = df_ml["daily_ridership"] / 1_000_000
    # Production recipe (xgb_recipe): early-stop on a validation tail held out of
    # the fit, then refit on the full training window with that tree count.
    xmodel, _ = fit_recipe(LIBRARY_DEFAULTS if xgb_params is None else xgb_params,
                           x_ml.iloc[:ml_cut], y_ml.iloc[:ml_cut])

    origins = forecast_origins(n, eval_days, horizon, step)
    logger.info(f"Rolling over {len(origins)} weekly origins (H={horizon})")
    exog_s = pd.DataFrame(scaler.transform(exog), index=y.index, columns=EXOG_COLS)
    s_all = sarimax_origin_forecasts(base, y, exog_s, train_end, origins, horizon)
    x_all = xgb_origin_forecasts(xmodel, df_ml, [y.index[p] for p in origins], horizon)
    a_all = [np.asarray(y.iloc[p + 1:p + 1 + horizon]) for p in origins]
    # Naive baselines from actuals known at each origin only
    sn_all = [seasonal_naive_path(y.iloc[:p + 1].to_numpy(), horizon) for p in origins]
    pe_all = [persistence_path(y.iloc[:p + 1].to_numpy(), horizon) for p in origins]

    return {
        "sarimax": s_all, "xgboost": x_all, "actual": a_all,
        "seasonal_naive": sn_all, "persistence": pe_all, "n_origins": len(origins),
    }


def summarize(blocks: dict, n_boot: int = 10000) -> dict:
    """Assemble the MAE/MAPE table, ensemble weight curve, and block-bootstrap
    significance from per-origin prediction blocks. Pure — no model fitting.
    """
    s = np.concatenate(blocks["sarimax"])
    x = np.concatenate(blocks["xgboost"])
    a = np.concatenate(blocks["actual"])
    sn = np.concatenate(blocks["seasonal_naive"])
    pe = np.concatenate(blocks["persistence"])

    sarimax_mae, xgb_mae = _mae(a, s), _mae(a, x)
    best_w, best_w_mae, curve = grid_search_weight(s, x, a)
    ens_50 = 0.5 * s + 0.5 * x

    # Bias (mean error, pred − actual): + = systematic over-forecast, − = under.
    # Invisible to MAE/MAPE but operationally critical for a staffing/planning tool,
    # and it reveals whether the ensemble cancels the individual models' skew.
    def _bias(pred):
        return float(np.mean(np.asarray(pred) - a))

    sn_mae = _mae(a, sn)
    # Relative MAE vs seasonal-naive (MASE-style; <1 beats naive). Scaled by the
    # seasonal-naive MAE on these same eval windows, not the in-sample one-step naive.
    def _rel(mae_val):
        return float(mae_val / sn_mae) if sn_mae else float("nan")

    # Fixed a-priori 50/50 ensemble blocks for significance — not the grid-search
    # best — so the ensemble is not credited for a weight tuned on this same data.
    ens50_blocks = [0.5 * sb + 0.5 * xb for sb, xb in zip(blocks["sarimax"], blocks["xgboost"], strict=True)]
    significance = {}
    for key, p1b, p2b, n1, n2 in [
        ("sarimax_vs_xgboost", blocks["sarimax"], blocks["xgboost"], "SARIMAX", "XGBoost"),
        ("ensemble_vs_xgboost", ens50_blocks, blocks["xgboost"], "ensemble", "XGBoost"),
        ("ensemble_vs_sarimax", ens50_blocks, blocks["sarimax"], "ensemble", "SARIMAX"),
    ]:
        mean_d, lo_ci, hi_ci = block_bootstrap_mae_diff(p1b, p2b, blocks["actual"], n_boot=n_boot)
        significance[key] = {
            "dmae": mean_d, "ci_lo": lo_ci, "ci_hi": hi_ci,
            "verdict": significance_verdict(lo_ci, hi_ci, n1, n2),
        }

    # Per-horizon ensemble MAE, averaged across origins (each 1..14-day step scored
    # once per origin). Pooling many origins cancels the date-specific noise that makes
    # any single forecast's per-day errors look non-monotonic, exposing the true
    # error-vs-horizon trend the thin live view can't show.
    ens_mat = np.vstack(ens50_blocks)          # (n_origins, horizon)
    a_mat = np.vstack(blocks["actual"])
    mae_by_horizon = [float(v) for v in np.abs(ens_mat - a_mat).mean(axis=0)]

    # Conformal band for the *ensemble* — the series production actually serves, so
    # it's calibrated on the configured weights, not the fixed 50/50 used above.
    served_blocks = [
        ENSEMBLE_SARIMAX_WEIGHT * sb + ENSEMBLE_XGB_WEIGHT * xb
        for sb, xb in zip(blocks["sarimax"], blocks["xgboost"], strict=True)
    ]
    conformal = conformal_half_widths(served_blocks, blocks["actual"], CI_COVERAGE)

    return {
        "n_origins": blocks["n_origins"],
        "n_points": int(len(a)),
        "mae": {
            "seasonal_naive": _mae(a, sn),
            "persistence": _mae(a, pe),
            "sarimax": sarimax_mae,
            "xgboost": xgb_mae,
            "ensemble_50_50": _mae(a, ens_50),
            "ensemble_best": best_w_mae,
            # At the configured weights — the monitoring baseline for the served line.
            "ensemble_served": _mae(a, np.concatenate(served_blocks)),
        },
        "mape": {
            "seasonal_naive": _mape(a, sn),
            "persistence": _mape(a, pe),
            "sarimax": _mape(a, s),
            "xgboost": _mape(a, x),
            "ensemble_50_50": _mape(a, ens_50),
        },
        "mase": {
            "seasonal_naive": 1.0,
            "persistence": _rel(_mae(a, pe)),
            "sarimax": _rel(sarimax_mae),
            "xgboost": _rel(xgb_mae),
            "ensemble_50_50": _rel(_mae(a, ens_50)),
        },
        "bias": {
            "seasonal_naive": _bias(sn),
            "persistence": _bias(pe),
            "sarimax": _bias(s),
            "xgboost": _bias(x),
            "ensemble_50_50": _bias(ens_50),
        },
        "best_weight": best_w,
        "best_weight_mae": best_w_mae,
        "weight_curve": curve,
        "significance": significance,
        "mae_by_horizon": mae_by_horizon,
        "conformal": conformal,
    }


def format_report(results: dict) -> str:
    lines = [
        "=" * 60,
        f"  14-DAY WALK-FORWARD  ({results['n_origins']} origins, {results['n_points']} points)",
        "=" * 60,
    ]
    labels = [
        ("seasonal-naive (m=7)", "seasonal_naive"),
        ("persistence (t-1)", "persistence"),
        ("SARIMAX", "sarimax"),
        ("XGBoost", "xgboost"),
        ("ensemble 50/50", "ensemble_50_50"),
    ]
    mase = results.get("mase", {})
    bias = results.get("bias", {})
    for name, key in labels:
        mae_v = results["mae"][key]
        mape_v = results["mape"].get(key)
        mape_s = f"  MAPE {mape_v:5.2f}%" if mape_v is not None else ""
        mase_s = f"  MASE {mase[key]:4.2f}" if key in mase else ""
        bias_s = f"  bias {bias[key]:+6.4f}" if key in bias else ""
        lines.append(f"  {name:<24} MAE {mae_v:6.4f}{mape_s}{mase_s}{bias_s}")
    bw = results["best_weight"]
    lines.append(f"  ensemble {bw:.2f}/{1 - bw:.2f} (BEST)  MAE {results['best_weight_mae']:6.4f}")
    lines.append("=" * 60)
    lines.append("Significance (block bootstrap, 95% CI on MAE difference):")
    for key, sig in results["significance"].items():
        lines.append(f"  {key:<22} dMAE={sig['dmae']:+.4f}  "
                     f"CI[{sig['ci_lo']:+.4f}, {sig['ci_hi']:+.4f}]  -> {sig['verdict']}")
    cf = results.get("conformal")
    if cf:
        hw = cf["half_width_by_horizon"]
        lines.append("=" * 60)
        lines.append(
            f"Conformal band ({cf['coverage']:.0%} nominal, n={cf['n_scores']} scores): "
            f"half-width day 1 +/-{hw[0]:.3f}M to day {len(hw)} +/-{hw[-1]:.3f}M"
        )
    return "\n".join(lines)


def _pinned_production_order() -> tuple | None:
    """Best-effort fetch of the SARIMAX order production is currently serving, from
    the S3 cache. Returns ``(order, seasonal_order)`` or ``None``.

    The backtest defaults to this so it evaluates the *same* SARIMAX architecture
    production ships, rather than a fresh auto_arima order that could silently
    diverge from the pinned one and make the dashboard's accuracy describe a model
    that never serves. Any failure (no credentials, cache miss) returns ``None`` and
    the caller falls back to re-searching — the prior behaviour — so local/offline
    runs still work.
    """
    try:
        cached = _load_cached_order(get_s3_client())
    except Exception as e:  # no creds / network — degrade to a fresh search
        logger.warning(f"Could not read pinned SARIMAX order ({e}); walk-forward will re-search")
        return None
    if cached is None:
        logger.warning("No pinned SARIMAX order in cache; walk-forward will re-search")
        return None
    order_pdq, seasonal_order, _ = cached
    return order_pdq, seasonal_order


def _serving_xgb_params() -> dict | None:
    """Best-effort fetch of production's tuned XGBoost parameters from S3, so the
    backtest evaluates the configuration that ships (as with the pinned SARIMAX
    order). Known leak, accepted for now: that tuning may have seen part of this
    backtest's window — the nested fix is to re-tune on the backtest's own training
    data."""
    try:
        return load_serving_params(get_s3_client())
    except Exception as e:
        logger.warning(f"Could not read XGBoost parameters ({e})")
        return None


def run(order: tuple | None = None) -> dict:
    logger.info(f"Regenerating ML gold; walk-forward H={H}, step={STEP}, eval window={EVAL_DAYS}d")
    preprocess_ml.run()

    df = pd.read_parquet(GOLD_SARIMA_LOCAL_PATH)
    df.index = pd.to_datetime(df.index)
    df = df.asfreq("D")
    y = df["daily_ridership"] / 1_000_000
    exog = df[EXOG_COLS]

    df_ml = pd.read_parquet(GOLD_ML_LOCAL_PATH)
    df_ml.index = pd.to_datetime(df_ml.index)
    df_ml = cast_categoricals(df_ml)

    # Pin the backtest to production's cached SARIMAX order so it evaluates the same
    # architecture that ships, not a freshly-searched order that could diverge from
    # it. Falls back to a fresh search when the pin is unavailable.
    if order is None:
        order = _pinned_production_order()
    logger.info(
        "Walk-forward SARIMAX order: "
        + (f"pinned to production {order[0]}x{order[1]}" if order else "re-searching (no pin available)")
    )

    xgb_params = _serving_xgb_params()
    if xgb_params is None:
        logger.warning("No persisted XGBoost parameters; walk-forward uses library defaults")
    blocks = walk_forward(y, exog, df_ml, order=order, xgb_params=xgb_params)
    results = summarize(blocks)
    results["xgboost_params_source"] = "serving" if xgb_params is not None else "library_defaults"
    results["sarimax_order_source"] = "pinned_production" if order else "re_searched"
    if order:
        results["sarimax_order"] = {"order": list(order[0]), "seasonal_order": list(order[1])}
    print(format_report(results))
    return results


if __name__ == "__main__":
    run()
