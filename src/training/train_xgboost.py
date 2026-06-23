from datetime import date

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from src.utils.config import (
    GOLD_ML_LOCAL_PATH,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    REPORTS_DIR,
    TEST_DAYS,
    XGBOOST_MODEL_NAME,
)
from src.utils.features import cast_categoricals, iterative_xgb_predict
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


def run() -> None:
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

    X_train, X_test = X.iloc[:-TEST_DAYS], X.iloc[-TEST_DAYS:]
    y_train, y_test = y.iloc[:-TEST_DAYS], y.iloc[-TEST_DAYS:]

    logger.info(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows | Features: {len(feature_cols)}")

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
        mlflow.log_params({**{k: v for k, v in XGB_PARAMS.items()}, "features": str(feature_cols)})
        mlflow.log_metrics({f"cv_fold_{i+1}_mae": m for i, m in enumerate(cv_maes)})
        mlflow.log_metric("cv_mean_mae", np.mean(cv_maes))

        # Early-stop on a validation slice carved from the TRAIN tail so the
        # reporting holdout (X_test) stays untouched. The tree count must not be
        # chosen by watching the same window the holdout MAE is computed on, since
        # the champion gate now trusts that logged MAE for promotion decisions.
        X_inner, X_val = X_train.iloc[:-TEST_DAYS], X_train.iloc[-TEST_DAYS:]
        y_inner, y_val = y_train.iloc[:-TEST_DAYS], y_train.iloc[-TEST_DAYS:]

        logger.info("Fitting evaluation XGBoost on inner-train split (early-stops on a held-out val slice)")
        eval_model = xgb.XGBRegressor(**XGB_PARAMS)
        eval_model.fit(
            X_inner, y_inner,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        # Iterative holdout to mirror production 14-day inference: predicted
        # ridership feeds back into the lag/rolling features rather than a one-shot
        # predict() that peeks at true lags the model won't have at serve time.
        test_start_idx = len(df) - TEST_DAYS
        y_pred = iterative_xgb_predict(eval_model, df, test_start_idx, TEST_DAYS)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mape = mean_absolute_percentage_error(y_test, y_pred)
        mlflow.log_metrics({"mae": mae, "rmse": rmse, "mape": mape})
        logger.info(f"Holdout metrics — MAE: {mae:.4f}M  RMSE: {rmse:.4f}M  MAPE: {mape:.2%}")

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(y_test.index, y_test.values, label="Actual", color="steelblue")
        ax.plot(y_test.index, y_pred, label="XGBoost forecast", linestyle="--", color="darkorange")
        ax.set_title(f"XGBoost Champion | MAE: {mae:.3f}M | MAPE: {mape:.2%}")
        ax.legend()
        plot_path = REPORTS_DIR / "xgboost_training_forecast.png"
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        mlflow.log_artifact(str(plot_path))

        # Per-day holdout predictions — the champion gate reads this to run the
        # ensemble weight analysis on a common, out-of-sample window for both
        # families (it cannot re-forecast the full-data production model honestly).
        holdout_df = pd.DataFrame({
            "date": y_test.index.strftime("%Y-%m-%d"),
            "y_true": y_test.to_numpy(),
            "y_pred": np.asarray(y_pred),
        })
        holdout_path = REPORTS_DIR / "holdout_predictions.json"
        holdout_df.to_json(holdout_path, orient="records")
        mlflow.log_artifact(str(holdout_path))

        # ── Production model: refit on ALL data (train + holdout) so the shipped
        #    model uses every observation up to the latest date. Early stopping
        #    chose the tree count against the holdout; the production fit reuses that
        #    fixed n_estimators on the full set (no holdout remains to early-stop on),
        #    which also stops the shipped model from being tuned on its own metric set.
        best_n = (eval_model.best_iteration or 0) + 1
        prod_params = {k: v for k, v in XGB_PARAMS.items() if k != "early_stopping_rounds"}
        prod_params["n_estimators"] = best_n
        mlflow.log_metric("best_n_estimators", best_n)
        logger.info(f"Refitting production XGBoost on full dataset (n_estimators={best_n})")
        final_model = xgb.XGBRegressor(**prod_params)
        final_model.fit(X, y, verbose=False)

        # SHAP runs before model registration below. Native categorical features
        # can trip up TreeExplainer on some shap/xgboost versions, so a failure
        # here must not abort the run and leave the model unregistered.
        try:
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
