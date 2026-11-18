from datetime import date

import mlflow
import mlflow.xgboost
import pandas as pd

from src.evaluation.origins import forecast_origins, origin_records, xgb_origin_forecasts
from src.training.holdout import log_holdout
from src.training.tune_xgboost import resolve_params
from src.training.xgb_recipe import FIXED_PARAMS, early_stop_n, fit_fixed
from src.utils.config import (
    FORECAST_HORIZON_DAYS,
    GOLD_ML_LOCAL_PATH,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    REPORTS_DIR,
    TEST_DAYS,
    XGBOOST_MODEL_NAME,
)
from src.utils.features import cast_categoricals
from src.utils.logger import get_logger
from src.utils.s3_helpers import get_s3_client

logger = get_logger(__name__)

TARGET_COL = "daily_ridership"


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c != TARGET_COL]


def run(gold_dvc_hash: str | None = None) -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading ML gold data")
    df = pd.read_parquet(GOLD_ML_LOCAL_PATH)
    df.index = pd.to_datetime(df.index)
    df = cast_categoricals(df)  # parquet does not preserve category dtype

    feature_cols = get_feature_cols(df)
    X = df[feature_cols]
    y = df[TARGET_COL] / 1_000_000

    X_train, y_train = X.iloc[:-TEST_DAYS], y.iloc[:-TEST_DAYS]

    logger.info(f"Train: {len(X_train)} rows | Test: {TEST_DAYS} rows | Features: {len(feature_cols)}")

    with mlflow.start_run(run_name="xgboost_training") as run:
        mlflow.set_tag("data_version_date", df.index.max().strftime("%Y-%m-%d"))
        mlflow.set_tag("dataset_row_count", len(df))
        mlflow.set_tag("project_phase", "champion_selection")
        mlflow.set_tag("run_date", str(date.today()))
        # Data lineage: pin this model version to the exact gold DVC hash it trained
        # on, so a model can be traced back to its data version from MLflow alone.
        if gold_dvc_hash:
            mlflow.set_tag("gold_dvc_hash", gold_dvc_hash)

        # Hyperparameters come from the tuning stage, which sees only pre-holdout
        # data, so the holdout below stays out-of-sample for the whole procedure.
        params, source, tuning = resolve_params(
            get_s3_client(), df.iloc[:-TEST_DAYS], X_train, y_train)
        mlflow.set_tag("params_source", source)
        mlflow.log_params({**params, **FIXED_PARAMS, "features": str(feature_cols)})
        if tuning is not None:
            mlflow.log_metrics({"tuning_validation_mae": tuning.validation_mae,
                                "tuning_baseline_mae": tuning.baseline_mae,
                                "tuning_best_trial_mae": tuning.best_trial_mae})
            mlflow.log_dict({**tuning.summary(), "trials": tuning.trials}, "xgboost_tuning.json")

        # Production recipe (xgb_recipe): the tree count is early-stopped on a slice
        # carved from the TRAIN tail — never on the holdout the gate scores — and the
        # eval model is refit on all pre-holdout data with that fixed count.
        logger.info("Early-stopping XGBoost on a validation slice of the training data")
        best_n = early_stop_n(params, X_train, y_train)
        mlflow.log_metric("best_n_estimators", best_n)

        logger.info(f"Fitting evaluation XGBoost on all pre-holdout data (n_estimators={best_n})")
        eval_model = fit_fixed(params, best_n, X_train, y_train)

        # Weekly 14-day forecasts over the holdout window, each iterative from the
        # actuals known at its origin: predicted ridership feeds back into the
        # lag/rolling features exactly as in production, and no forecast runs past
        # the 14-day horizon production serves.
        origins = forecast_origins(len(df), TEST_DAYS, FORECAST_HORIZON_DAYS)
        origin_dates = [df.index[p] for p in origins]
        preds = xgb_origin_forecasts(eval_model, df, origin_dates, FORECAST_HORIZON_DAYS)
        log_holdout(origin_records(origin_dates, preds, y), "XGBoost")

        # ── Production model: refit on ALL data (train + holdout) with the same
        #    fixed tree count, so the shipped model uses every observation up to the
        #    latest date (no holdout remains to early-stop on).
        logger.info(f"Refitting production XGBoost on full dataset (n_estimators={best_n})")
        final_model = fit_fixed(params, best_n, X, y)

        # SHAP runs before model registration below. Native categorical features
        # can trip up TreeExplainer on some shap/xgboost versions, so a failure
        # here must not abort the run and leave the model unregistered.
        try:
            import matplotlib.pyplot as plt
            import shap
            logger.info("Computing SHAP values on full dataset")
            explainer = shap.TreeExplainer(final_model)
            shap_values = explainer.shap_values(X)

            _FEATURE_LABELS = {
                "temp":            "Temperature (°C)",
                "precip":          "Precipitation (mm)",
                "snow":            "Snow (mm)",
                "is_holiday":      "Holiday",
                "snow_lag1":       "Snow Lag 1 (mm)",
                "day_of_week":     "Day of Week",
                "month":           "Month",
                "is_weekend":      "Weekend",
                "ridership_lag1":  "Ridership Lag 1 (M)",
                "ridership_lag2":  "Ridership Lag 2 (M)",
                "ridership_lag3":  "Ridership Lag 3 (M)",
                "ridership_lag7":  "Ridership Lag 7 (M)",
                "ridership_lag14": "Ridership Lag 14 (M)",
                "ridership_14d_avg": "14-Day Avg Ridership (M)",
                "ridership_7d_std":  "7-Day Std Ridership (M)",
                "precip_lag1":     "Precipitation Lag 1 (mm)",
                "temp_lag1":       "Temperature Lag 1 (°C)",
            }
            X_labeled = X.rename(columns=_FEATURE_LABELS)

            shap_path = REPORTS_DIR / "xgboost_shap_summary.png"
            with plt.style.context("dark_background"):
                fig_shap, _ = plt.subplots(figsize=(10, 7))
                fig_shap.patch.set_facecolor("#0e1117")
                shap.summary_plot(shap_values, X_labeled, show=False)
                fig = plt.gcf()
                fig.set_facecolor("#0e1117")
                for ax in fig.get_axes():
                    ax.set_facecolor("#0e1117")
                    ax.tick_params(colors="white", labelcolor="white")
                    ax.xaxis.label.set_color("white")
                    ax.yaxis.label.set_color("white")
                    for text in ax.get_xticklabels() + ax.get_yticklabels():
                        text.set_color("white")
                    for spine in ax.spines.values():
                        spine.set_edgecolor("white")
                for text in fig.findobj(plt.Text):
                    text.set_color("white")
                fig_shap.savefig(shap_path, dpi=150, bbox_inches="tight", facecolor="#0e1117")
                plt.close(fig_shap)
            mlflow.log_artifact(str(shap_path))
        except Exception as e:
            logger.warning(f"SHAP computation failed, skipping SHAP artifact: {e}")

        mlflow.xgboost.log_model(
            final_model, "xgboost_model",
            registered_model_name=XGBOOST_MODEL_NAME,
        )
        logger.info(f"Model registered: {XGBOOST_MODEL_NAME} (run_id={run.info.run_id})")


if __name__ == "__main__":
    run()
