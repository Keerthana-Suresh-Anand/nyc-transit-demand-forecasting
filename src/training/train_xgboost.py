from datetime import date

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

from src.evaluation.origins import forecast_origins, origin_records, xgb_origin_forecasts
from src.training.holdout import log_holdout
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

logger = get_logger(__name__)

TARGET_COL = "daily_ridership"
CV_SPLITS = 5

XGB_PARAMS = {
    "n_estimators": 500,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "early_stopping_rounds": 50,
    "eval_metric": "mae",
    "random_state": 42,
    "enable_categorical": True,
}


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

    tscv = TimeSeriesSplit(n_splits=CV_SPLITS)
    cv_maes = []

    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train)):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

        model_cv = xgb.XGBRegressor(**XGB_PARAMS)
        model_cv.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        fold_mae = mean_absolute_error(y_val, model_cv.predict(X_val))
        cv_maes.append(fold_mae)
        logger.info(f"CV fold {fold + 1}/{CV_SPLITS} MAE: {fold_mae:.4f}M")

    logger.info(f"CV mean MAE: {np.mean(cv_maes):.4f}M (±{np.std(cv_maes):.4f})")

    with mlflow.start_run(run_name="xgboost_training") as run:
        mlflow.set_tag("data_version_date", df.index.max().strftime("%Y-%m-%d"))
        mlflow.set_tag("dataset_row_count", len(df))
        mlflow.set_tag("project_phase", "champion_selection")
        mlflow.set_tag("run_date", str(date.today()))
        # Data lineage: pin this model version to the exact gold DVC hash it trained
        # on, so a model can be traced back to its data version from MLflow alone.
        if gold_dvc_hash:
            mlflow.set_tag("gold_dvc_hash", gold_dvc_hash)
        mlflow.log_params({**{k: v for k, v in XGB_PARAMS.items()}, "features": str(feature_cols)})
        mlflow.log_metrics({f"cv_fold_{i+1}_mae": m for i, m in enumerate(cv_maes)})
        mlflow.log_metric("cv_mean_mae", np.mean(cv_maes))

        # Same recipe as the production fit below (and the walk-forward): early-stop
        # on a validation slice carved from the TRAIN tail to choose the tree count,
        # then refit on all pre-holdout data with that fixed count. The tree count
        # must not be chosen by watching the holdout the gate scores, and the eval
        # model must not lose the month before the holdout to the validation slice.
        X_inner, X_val = X_train.iloc[:-TEST_DAYS], X_train.iloc[-TEST_DAYS:]
        y_inner, y_val = y_train.iloc[:-TEST_DAYS], y_train.iloc[-TEST_DAYS:]

        logger.info("Early-stopping XGBoost on a validation slice of the training data")
        es_model = xgb.XGBRegressor(**XGB_PARAMS)
        es_model.fit(X_inner, y_inner, eval_set=[(X_val, y_val)], verbose=False)
        best_n = (es_model.best_iteration or 0) + 1
        fixed_params = {k: v for k, v in XGB_PARAMS.items() if k != "early_stopping_rounds"}
        fixed_params["n_estimators"] = best_n
        mlflow.log_metric("best_n_estimators", best_n)

        logger.info(f"Fitting evaluation XGBoost on all pre-holdout data (n_estimators={best_n})")
        eval_model = xgb.XGBRegressor(**fixed_params).fit(X_train, y_train, verbose=False)

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
        final_model = xgb.XGBRegressor(**fixed_params)
        final_model.fit(X, y, verbose=False)

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
