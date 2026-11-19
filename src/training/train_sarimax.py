import json
import pickle
import warnings
from datetime import date

import mlflow
import mlflow.statsmodels
import numpy as np
import pandas as pd
import pmdarima as pm
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.evaluation.origins import forecast_origins, origin_records, sarimax_origin_forecasts
from src.training.holdout import log_holdout
from src.utils.config import (
    FORECAST_HORIZON_DAYS,
    GOLD_SARIMA_LOCAL_PATH,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    REPORTS_DIR,
    S3_SARIMAX_ORDER_KEY,
    SARIMAX_EXOG_COLS,
    SARIMAX_MODEL_NAME,
    SARIMAX_ORDER_VAL_DAYS,
    SARIMAX_RESEARCH_DAYS,
    TEST_DAYS,
)
from src.utils.logger import get_logger
from src.utils.s3_helpers import get_s3_client, read_s3_json, write_s3_json

warnings.filterwarnings("ignore")
# Keep convergence warnings visible — a non-converged SARIMAX is a real signal,
# not noise to silence with the blanket filter above.
warnings.filterwarnings("always", category=ConvergenceWarning)
logger = get_logger(__name__)

EXOG_COLS = SARIMAX_EXOG_COLS


def load_data() -> tuple[pd.Series, pd.DataFrame]:
    df = pd.read_parquet(GOLD_SARIMA_LOCAL_PATH)
    df.index = pd.to_datetime(df.index)
    df = df.asfreq("D")
    y = df["daily_ridership"] / 1_000_000
    return y, df[EXOG_COLS]


def scale_exog(
    exog: pd.DataFrame, train_idx: pd.Index, test_idx: pd.Index
) -> tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler]:
    scaler = MinMaxScaler()
    train_exog = pd.DataFrame(
        scaler.fit_transform(exog.loc[train_idx]), index=train_idx, columns=EXOG_COLS
    )
    test_exog = pd.DataFrame(
        scaler.transform(exog.loc[test_idx]), index=test_idx, columns=EXOG_COLS
    )
    return train_exog, test_exog, scaler


def find_best_params(train_y: pd.Series, train_exog: pd.DataFrame) -> tuple:
    logger.info("Starting Auto-ARIMA stepwise search...")
    # pmdarima >= 2.0 names the exogenous argument `X`. An unknown keyword such as
    # `exog=` is silently swallowed into **fit_args, so the search would pick an
    # order for a model with no regressors.
    auto_model = pm.auto_arima(
        train_y,
        X=train_exog,
        start_p=0,
        start_q=0,
        max_p=3,
        max_q=3,
        m=7,
        seasonal=True,
        start_P=0,
        D=1,
        error_action="ignore",
        suppress_warnings=True,
        stepwise=True,
    )
    logger.info(f"Best model found: SARIMAX{auto_model.order}x{auto_model.seasonal_order}")
    return auto_model.order, auto_model.seasonal_order


def _load_cached_order(s3) -> tuple[tuple, tuple, date] | None:
    """Return (order, seasonal_order, search_date) from the S3 cache, or None."""
    try:
        data = read_s3_json(s3, S3_SARIMAX_ORDER_KEY)
        return (
            tuple(data["order"]),
            tuple(data["seasonal_order"]),
            date.fromisoformat(data["search_date"]),
        )
    except Exception:
        return None


def _save_cached_order(s3, order: tuple, seasonal_order: tuple, evidence: dict | None = None) -> None:
    try:
        write_s3_json(
            s3,
            {
                "order": list(order),
                "seasonal_order": list(seasonal_order),
                "search_date": str(date.today()),
                "validation_mae": evidence or {},
            },
            S3_SARIMAX_ORDER_KEY,
        )
        logger.info(f"Pinned SARIMAX order to S3: {order}x{seasonal_order}")
    except Exception as e:
        logger.warning(f"Could not persist SARIMAX order to S3: {e}")


def select_order_by_validation(
    train_y: pd.Series, train_exog: pd.DataFrame, candidates: list[tuple[tuple, tuple]]
) -> tuple[tuple[tuple, tuple], dict[str, float]]:
    """Pick the candidate order with the lowest 14-day rolling-origin MAE over the
    last SARIMAX_ORDER_VAL_DAYS of the training data.

    auto_arima ranks orders by in-sample AIC, which is not what the model is used
    for: on 2026-11 data its exog-aware pick, (0,0,1)x(1,1,1,7), had a walk-forward
    MAE of 0.265 vs 0.204 for the order it would have replaced. Out-of-sample
    forecast error is the criterion that matters, and it is measured on training
    data only, so the holdout stays untouched. Candidates that fail to fit are
    skipped; returns (best, {candidate: MAE}).
    """
    n = len(train_y)
    fit_end = n - SARIMAX_ORDER_VAL_DAYS
    origins = forecast_origins(n, SARIMAX_ORDER_VAL_DAYS, FORECAST_HORIZON_DAYS)
    actual = np.concatenate(
        [train_y.iloc[p + 1:p + 1 + FORECAST_HORIZON_DAYS].to_numpy() for p in origins])
    scores: dict[str, float] = {}
    for order, seasonal_order in candidates:
        label = f"{tuple(order)}x{tuple(seasonal_order)}"
        try:
            res = SARIMAX(train_y.iloc[:fit_end], exog=train_exog.iloc[:fit_end], order=order,
                          seasonal_order=seasonal_order, enforce_stationarity=False,
                          enforce_invertibility=False).fit(disp=False)
            preds = sarimax_origin_forecasts(res, train_y, train_exog, fit_end, origins,
                                             FORECAST_HORIZON_DAYS)
        except Exception as e:
            logger.warning(f"Order {label} failed validation fit: {e}")
            continue
        scores[label] = float(np.mean(np.abs(np.concatenate(preds) - actual)))
        logger.info(f"Order {label}: validation MAE {scores[label]:.4f}M "
                    f"({len(origins)} weekly 14-day forecasts)")
    if not scores:
        raise RuntimeError("No SARIMAX order candidate could be fit for validation")
    best = min(scores, key=scores.get)
    return candidates[list(f"{tuple(o)}x{tuple(s)}" for o, s in candidates).index(best)], scores


def resolve_order(s3, train_y: pd.Series, train_exog: pd.DataFrame) -> tuple[tuple, tuple, str]:
    """Reuse the cached SARIMAX order while it is fresh; otherwise re-select and
    re-pin it. Keeps the production architecture stable run to run instead of
    letting the order change every training cycle.

    Re-selection: auto_arima proposes an order, and the expiring cached order
    competes with it on out-of-sample validation error (``select_order_by_validation``).
    """
    cached = _load_cached_order(s3)
    if cached is not None:
        order, seasonal_order, searched = cached
        age = (date.today() - searched).days
        if age < SARIMAX_RESEARCH_DAYS:
            logger.info(
                f"Reusing cached SARIMAX order {order}x{seasonal_order} "
                f"(searched {searched}, {age}d ago < {SARIMAX_RESEARCH_DAYS}d)"
            )
            return order, seasonal_order, "cached"
        logger.info(
            f"Cached SARIMAX order is {age}d old (≥ {SARIMAX_RESEARCH_DAYS}d) — re-selecting"
        )

    proposal = tuple(find_best_params(train_y, train_exog))
    candidates = [proposal]
    if cached is not None and tuple(cached[:2]) != proposal:
        candidates.append(tuple(cached[:2]))
    if len(candidates) == 1:
        _save_cached_order(s3, *proposal)
        return *proposal, "auto_arima_stepwise"
    (order, seasonal_order), scores = select_order_by_validation(train_y, train_exog, candidates)
    _save_cached_order(s3, order, seasonal_order, scores)
    return order, seasonal_order, "validated"


def run(gold_dvc_hash: str | None = None) -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading SARIMA gold data")
    y, exog = load_data()

    split_date = y.index[-TEST_DAYS]
    train_idx = y.loc[: split_date - pd.Timedelta(days=1)].index
    test_idx = y.loc[split_date:].index
    train_y = y.loc[train_idx]

    train_exog, _, eval_scaler = scale_exog(exog, train_idx, test_idx)

    s3 = get_s3_client()
    best_order, best_seasonal, order_source = resolve_order(s3, train_y, train_exog)

    with mlflow.start_run(run_name="sarimax_training") as run:
        mlflow.set_tag("data_version_date", y.index.max().strftime("%Y-%m-%d"))
        mlflow.set_tag("dataset_row_count", len(y))
        mlflow.set_tag("project_phase", "champion_selection")
        mlflow.set_tag("run_date", str(date.today()))
        # Data lineage: pin this model version to the exact gold DVC hash it trained
        # on, so a model can be traced back to its data version from MLflow alone.
        if gold_dvc_hash:
            mlflow.set_tag("gold_dvc_hash", gold_dvc_hash)
        mlflow.log_params(
            {
                "order": str(best_order),
                "seasonal_order": str(best_seasonal),
                "exog_cols": str(EXOG_COLS),
                "test_days": TEST_DAYS,
                "search_method": order_source,
            }
        )

        # ── Holdout evaluation: fit on the data before the last TEST_DAYS only, then
        #    forecast 14 days from each weekly origin in that window, re-anchoring on
        #    the actuals known at the origin — the task production serves.
        logger.info("Fitting evaluation SARIMAX on training split for holdout metrics")
        eval_model = SARIMAX(
            train_y,
            exog=train_exog,
            order=best_order,
            seasonal_order=best_seasonal,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)

        exog_scaled = pd.DataFrame(eval_scaler.transform(exog), index=exog.index, columns=EXOG_COLS)
        origins = forecast_origins(len(y), TEST_DAYS, FORECAST_HORIZON_DAYS)
        preds = sarimax_origin_forecasts(
            eval_model, y, exog_scaled, len(train_y), origins, FORECAST_HORIZON_DAYS
        )
        log_holdout(origin_records([y.index[p] for p in origins], preds, y), "SARIMAX")

        # ── Production model: refit on ALL data (train + holdout) with the chosen
        #    order so the shipped model uses every observation up to the latest
        #    date. The scaler is refit on the full exog for the same reason — at
        #    inference time exog must be scaled by a scaler that saw the same data
        #    the production model trained on, or the scaling will be inconsistent.
        logger.info("Refitting production SARIMAX + scaler on full dataset")
        prod_scaler = MinMaxScaler()
        full_exog = pd.DataFrame(
            prod_scaler.fit_transform(exog), index=exog.index, columns=EXOG_COLS
        )
        prod_model = SARIMAX(
            y,
            exog=full_exog,
            order=best_order,
            seasonal_order=best_seasonal,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)
        if getattr(prod_model, "mle_retvals", {}).get("converged") is False:
            logger.warning("Production SARIMAX did NOT converge — check exog scaling / order")
        mlflow.log_metric("aic", prod_model.aic)

        scaler_path = REPORTS_DIR / "sarimax_scaler.pkl"
        with open(scaler_path, "wb") as f:
            pickle.dump(prod_scaler, f)
        mlflow.log_artifact(str(scaler_path))
        logger.info("Full-data scaler logged as MLflow artifact")

        # Exog coefficients for the dashboard, read from the production (full-data)
        # model so the panel matches the model that actually serves forecasts. Exog
        # were MinMax-scaled before fitting, so the coefficient magnitudes are
        # directly comparable across features. p-values flag significant effects.
        coef_records = [
            {
                "feature": col,
                "coefficient": float(prod_model.params[col]),
                "p_value": float(prod_model.pvalues[col]),
            }
            for col in EXOG_COLS
            if col in prod_model.params.index
        ]
        coef_path = REPORTS_DIR / "sarimax_coefficients.json"
        with open(coef_path, "w") as f:
            json.dump(
                {
                    "exog_coefficients": coef_records,
                    "note": "Coefficients on MinMax-scaled exogenous features; magnitudes are comparable.",
                    "run_date": str(date.today()),
                },
                f,
                indent=2,
            )
        mlflow.log_artifact(str(coef_path))
        logger.info(f"SARIMAX exog coefficients written: {coef_records}")

        mlflow.statsmodels.log_model(
            prod_model,
            "sarimax_model",
            registered_model_name=SARIMAX_MODEL_NAME,
        )
        logger.info(f"Model registered: {SARIMAX_MODEL_NAME} (run_id={run.info.run_id})")


if __name__ == "__main__":
    run()
