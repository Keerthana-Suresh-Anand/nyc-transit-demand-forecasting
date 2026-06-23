"""14-day rolling-origin walk-forward evaluation + block-bootstrap significance.

This is the evidence behind the 50/50 ensemble weight and the "keep both models"
decision (see CLAUDE.md → "Model evaluation methodology"). It evaluates SARIMAX,
XGBoost, naive baselines, and the ensemble the way production actually serves
them: a 14-day forecast re-anchored weekly. Models are trained once on data up to
the eval window; at each weekly origin SARIMAX state is updated with observed
actuals (`.append`, params fixed) and XGBoost forecasts 14 days iteratively,
seeding lags from actuals known at that origin. This removes the unfair 60-day
compounding that a single long holdout imposes on XGBoost.

Read-only w.r.t. MLflow and S3 — an analysis/reproducibility tool, run on demand,
not part of the scheduled pipelines. The statistical core (`block_bootstrap_mae_diff`,
`summarize`) is pure and unit-tested; `run()` does the heavy model fitting.
"""
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.evaluation.evaluate_models import grid_search_weight
from src.training.train_sarimax import EXOG_COLS, find_best_params
from src.training.train_xgboost import XGB_PARAMS, get_feature_cols
from src.transformation import preprocess_ml
from src.utils.config import GOLD_ML_LOCAL_PATH, GOLD_SARIMA_LOCAL_PATH
from src.utils.features import cast_categoricals, iterative_xgb_predict
from src.utils.logger import get_logger

warnings.filterwarnings("ignore")
warnings.filterwarnings("always", category=ConvergenceWarning)  # keep convergence signal visible
logger = get_logger(__name__)

EVAL_DAYS = 90   # window over which origins roll
STEP = 7         # weekly re-anchor, matching production cadence
H = 14           # forecast horizon, matching production


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


def significance_verdict(lo: float, hi: float, p1: str, p2: str) -> str:
    if hi < 0:
        return f"{p1} significantly better"
    if lo > 0:
        return f"{p2} significantly better"
    return "TIE (95% CI includes 0)"


def walk_forward(
    y: pd.Series, exog: pd.DataFrame, df_ml: pd.DataFrame,
    *, eval_days: int = EVAL_DAYS, step: int = STEP, horizon: int = H, order=None,
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
    xmodel = xgb.XGBRegressor(**XGB_PARAMS)
    xmodel.fit(x_ml.iloc[:ml_cut], y_ml.iloc[:ml_cut],
               eval_set=[(x_ml.iloc[ml_cut - 30:ml_cut], y_ml.iloc[ml_cut - 30:ml_cut])], verbose=False)

    s_all, x_all, a_all, sn_all, pe_all = [], [], [], [], []
    origins = list(range(train_end - 1, n - horizon, step))
    logger.info(f"Rolling over {len(origins)} weekly origins (H={horizon})")
    for p in origins:
        if p >= train_end:
            add_idx = y.index[train_end:p + 1]
            add_exog = pd.DataFrame(scaler.transform(exog.iloc[train_end:p + 1]),
                                    index=add_idx, columns=EXOG_COLS)
            res = base.append(y.iloc[train_end:p + 1], exog=add_exog, refit=False)
        else:
            res = base
        fut_exog = pd.DataFrame(scaler.transform(exog.iloc[p + 1:p + 1 + horizon]),
                                index=y.index[p + 1:p + 1 + horizon], columns=EXOG_COLS)
        s_pred = np.asarray(res.get_forecast(steps=horizon, exog=fut_exog).predicted_mean)

        pos_ml = df_ml.index.get_loc(y.index[p])
        x_pred = iterative_xgb_predict(xmodel, df_ml, pos_ml + 1, horizon)

        actual = np.asarray(y.iloc[p + 1:p + 1 + horizon])
        # Multi-step naive baselines using only actuals known at origin p
        naive = np.array([y.iloc[p + (h - 7 if h <= 7 else h - 14)] for h in range(1, horizon + 1)])
        pers = np.full(horizon, y.iloc[p])

        s_all.append(s_pred)
        x_all.append(x_pred)
        a_all.append(actual)
        sn_all.append(naive)
        pe_all.append(pers)

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

    # Fixed a-priori 50/50 ensemble blocks for significance — not the grid-search
    # best — so the ensemble is not credited for a weight tuned on this same data.
    ens50_blocks = [0.5 * sb + 0.5 * xb for sb, xb in zip(blocks["sarimax"], blocks["xgboost"])]
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
        },
        "mape": {
            "seasonal_naive": _mape(a, sn),
            "persistence": _mape(a, pe),
            "sarimax": _mape(a, s),
            "xgboost": _mape(a, x),
            "ensemble_50_50": _mape(a, ens_50),
        },
        "best_weight": best_w,
        "best_weight_mae": best_w_mae,
        "weight_curve": curve,
        "significance": significance,
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
    for name, key in labels:
        mae_v = results["mae"][key]
        mape_v = results["mape"].get(key)
        mape_s = f"   MAPE {mape_v:5.2f}%" if mape_v is not None else ""
        lines.append(f"  {name:<24} MAE {mae_v:6.4f}{mape_s}")
    bw = results["best_weight"]
    lines.append(f"  ensemble {bw:.2f}/{1 - bw:.2f} (BEST)  MAE {results['best_weight_mae']:6.4f}")
    lines.append("=" * 60)
    lines.append("Significance (block bootstrap, 95% CI on MAE difference):")
    for key, sig in results["significance"].items():
        lines.append(f"  {key:<22} dMAE={sig['dmae']:+.4f}  "
                     f"CI[{sig['ci_lo']:+.4f}, {sig['ci_hi']:+.4f}]  -> {sig['verdict']}")
    return "\n".join(lines)


def run() -> dict:
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

    blocks = walk_forward(y, exog, df_ml)
    results = summarize(blocks)
    print(format_report(results))
    return results


if __name__ == "__main__":
    run()
