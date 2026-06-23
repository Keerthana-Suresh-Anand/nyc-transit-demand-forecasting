import json
import pickle
import warnings
from datetime import date

import mlflow
import mlflow.statsmodels
import numpy as np
import pandas as pd
import pmdarima as pm
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.utils.config import (
    GOLD_SARIMA_LOCAL_PATH,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    REPORTS_DIR,
    S3_SARIMAX_ORDER_KEY,
    SARIMAX_EXOG_COLS,
    SARIMAX_MODEL_NAME,
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
    auto_model = pm.auto_arima(
        train_y, exog=train_exog,
        start_p=0, start_q=0, max_p=3, max_q=3,
        m=7, seasonal=True, start_P=0, D=1,
        error_action="ignore", suppress_warnings=True, stepwise=True,
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


def _save_cached_order(s3, order: tuple, seasonal_order: tuple) -> None:
    try:
        write_s3_json(s3, {
            "order": list(order),
            "seasonal_order": list(seasonal_order),
            "search_date": str(date.today()),
        }, S3_SARIMAX_ORDER_KEY)
        logger.info(f"Pinned SARIMAX order to S3: {order}x{seasonal_order}")
    except Exception as e:
        logger.warning(f"Could not persist SARIMAX order to S3: {e}")


def resolve_order(s3, train_y: pd.Series, train_exog: pd.DataFrame) -> tuple[tuple, tuple, str]:
    """Reuse the cached SARIMAX order while it is fresh; otherwise re-search and
    re-pin it. Keeps the production architecture stable run to run instead of
    letting auto_arima silently pick a new order every training cycle.
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
        logger.info(f"Cached SARIMAX order is {age}d old (≥ {SARIMAX_RESEARCH_DAYS}d) — re-searching")

    order, seasonal_order = find_best_params(train_y, train_exog)
    _save_cached_order(s3, order, seasonal_order)
    return order, seasonal_order, "auto_arima_stepwise"


def run() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading SARIMA gold data")
    y, exog = load_data()

    split_date = y.index[-TEST_DAYS]
    train_idx = y.loc[: split_date - pd.Timedelta(days=1)].index
    test_idx = y.loc[split_date:].index
    train_y, test_y = y.loc[train_idx], y.loc[test_idx]

    train_exog, test_exog, _ = scale_exog(exog, train_idx, test_idx)

    s3 = get_s3_client()
    best_order, best_seasonal, order_source = resolve_order(s3, train_y, train_exog)

    with mlflow.start_run(run_name="sarimax_training") as run:
        mlflow.set_tag("data_version_date", y.index.max().strftime("%Y-%m-%d"))
        mlflow.set_tag("dataset_row_count", len(y))
        mlflow.set_tag("project_phase", "champion_selection")
        mlflow.set_tag("run_date", str(date.today()))
        mlflow.log_params({
            "order": str(best_order),
            "seasonal_order": str(best_seasonal),
            "exog_cols": str(EXOG_COLS),
            "test_days": TEST_DAYS,
            "search_method": order_source,
        })

        # ── Holdout evaluation: fit on the training split only and score the
        #    untouched last TEST_DAYS so the logged metrics reflect genuine
        #    out-of-sample performance.
        logger.info("Fitting evaluation SARIMAX on training split for holdout metrics")
        eval_model = SARIMAX(
            train_y, exog=train_exog,
            order=best_order, seasonal_order=best_seasonal,
            enforce_stationarity=False, enforce_invertibility=False,
        ).fit(disp=False)

        forecast = eval_model.get_forecast(steps=len(test_y), exog=test_exog)
        y_pred = forecast.predicted_mean
        conf_int = forecast.conf_int()

        mae = mean_absolute_error(test_y, y_pred)
        rmse = np.sqrt(mean_squared_error(test_y, y_pred))
        mape = mean_absolute_percentage_error(test_y, y_pred)
        mlflow.log_metrics({"mae": mae, "rmse": rmse, "mape": mape})
        logger.info(f"Holdout metrics — MAE: {mae:.4f}M  RMSE: {rmse:.4f}M  MAPE: {mape:.2%}")

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(test_y.index, test_y, label="Actual", color="steelblue")
        ax.plot(test_y.index, y_pred, label="SARIMAX forecast", linestyle="--", color="darkorange")
        ax.fill_between(test_y.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1],
                        color="gray", alpha=0.25, label="95% CI")
        ax.set_title(f"SARIMAX Champion | MAE: {mae:.3f}M | MAPE: {mape:.2%}")
        ax.legend()
        plot_path = REPORTS_DIR / "sarimax_training_forecast.png"
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        mlflow.log_artifact(str(plot_path))

        # Per-day holdout predictions — the champion gate reads this to run the
        # ensemble weight analysis on a common, out-of-sample window for both
        # families (it cannot re-forecast the full-data production model honestly).
        holdout_df = pd.DataFrame({
            "date": test_y.index.strftime("%Y-%m-%d"),
            "y_true": test_y.to_numpy(),
            "y_pred": np.asarray(y_pred),
        })
        holdout_path = REPORTS_DIR / "holdout_predictions.json"
        holdout_df.to_json(holdout_path, orient="records")
        mlflow.log_artifact(str(holdout_path))

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
            y, exog=full_exog,
            order=best_order, seasonal_order=best_seasonal,
            enforce_stationarity=False, enforce_invertibility=False,
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
            for col in EXOG_COLS if col in prod_model.params.index
        ]
        coef_path = REPORTS_DIR / "sarimax_coefficients.json"
        with open(coef_path, "w") as f:
            json.dump({
                "exog_coefficients": coef_records,
                "note": "Coefficients on MinMax-scaled exogenous features; magnitudes are comparable.",
                "run_date": str(date.today()),
            }, f, indent=2)
        mlflow.log_artifact(str(coef_path))
        logger.info(f"SARIMAX exog coefficients written: {coef_records}")

        mlflow.statsmodels.log_model(
            prod_model, "sarimax_model",
            registered_model_name=SARIMAX_MODEL_NAME,
        )
        logger.info(f"Model registered: {SARIMAX_MODEL_NAME} (run_id={run.info.run_id})")


if __name__ == "__main__":
    run()
