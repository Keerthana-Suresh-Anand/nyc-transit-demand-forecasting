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
    XGBOOST_MODEL_NAME,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

TARGET_COL = "daily_ridership"
TEST_DAYS = 30
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

    with mlflow.start_run(run_name="xgboost_champion") as run:
        mlflow.set_tag("data_version_date", df.index.max().strftime("%Y-%m-%d"))
        mlflow.set_tag("dataset_row_count", len(df))
        mlflow.set_tag("project_phase", "champion_selection")
        mlflow.set_tag("run_date", str(date.today()))
        mlflow.log_params({**{k: v for k, v in XGB_PARAMS.items()}, "features": str(feature_cols)})
        mlflow.log_metrics({f"cv_fold_{i+1}_mae": m for i, m in enumerate(cv_maes)})
        mlflow.log_metric("cv_mean_mae", np.mean(cv_maes))

        logger.info("Training final XGBoost model on full training set")
        final_model = xgb.XGBRegressor(**XGB_PARAMS)
        final_model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        y_pred = final_model.predict(X_test)
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
        plot_path = REPORTS_DIR / "xgboost_champion_forecast.png"
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        mlflow.log_artifact(str(plot_path))

        import shap
        logger.info("Computing SHAP values")
        explainer = shap.TreeExplainer(final_model)
        shap_values = explainer.shap_values(X_test)

        fig_shap, ax_shap = plt.subplots(figsize=(10, 7))
        shap.summary_plot(shap_values, X_test, show=False)
        shap_path = REPORTS_DIR / "xgboost_shap_summary.png"
        fig_shap.savefig(shap_path, dpi=150, bbox_inches="tight")
        plt.close(fig_shap)
        mlflow.log_artifact(str(shap_path))

        mlflow.xgboost.log_model(
            final_model, "xgboost_model",
            registered_model_name=XGBOOST_MODEL_NAME,
        )
        logger.info(f"Model registered: {XGBOOST_MODEL_NAME} (run_id={run.info.run_id})")


if __name__ == "__main__":
    run()
