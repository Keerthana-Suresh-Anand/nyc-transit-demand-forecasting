"""
Loads SARIMAX and XGBoost champions from MLflow Registry, runs both on the same
holdout set, and promotes the winner to 'Production'. The loser goes to 'Staging'.
"""
import warnings
from datetime import date

import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from src.utils.config import (
    GOLD_ML_LOCAL_PATH, GOLD_SARIMA_LOCAL_PATH,
    MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI,
    SARIMAX_MODEL_NAME, XGBOOST_MODEL_NAME,
)
from src.utils.logger import get_logger

warnings.filterwarnings("ignore")
logger = get_logger(__name__)

TEST_DAYS = 30
SARIMAX_EXOG_COLS = ["temp", "precip", "snow_lag1", "is_holiday"]


def evaluate_sarimax(client: MlflowClient) -> tuple[float, float, float]:
    df = pd.read_parquet(GOLD_SARIMA_LOCAL_PATH)
    df.index = pd.to_datetime(df.index)
    df = df.asfreq("D")
    y = df["daily_ridership"] / 1_000_000

    train_idx = y.iloc[:-TEST_DAYS].index
    test_idx = y.iloc[-TEST_DAYS:].index
    train_y, test_y = y.loc[train_idx], y.loc[test_idx]

    scaler = MinMaxScaler()
    train_exog = pd.DataFrame(
        scaler.fit_transform(df.loc[train_idx, SARIMAX_EXOG_COLS]),
        index=train_idx, columns=SARIMAX_EXOG_COLS,
    )
    test_exog = pd.DataFrame(
        scaler.transform(df.loc[test_idx, SARIMAX_EXOG_COLS]),
        index=test_idx, columns=SARIMAX_EXOG_COLS,
    )

    model_uri = f"models:/{SARIMAX_MODEL_NAME}/Production"
    model = mlflow.statsmodels.load_model(model_uri)
    y_pred = model.get_forecast(steps=len(test_y), exog=test_exog).predicted_mean

    mae = mean_absolute_error(test_y, y_pred)
    rmse = np.sqrt(mean_squared_error(test_y, y_pred))
    mape = mean_absolute_percentage_error(test_y, y_pred)
    mse = float(np.mean(np.array(y_pred) - np.array(test_y)))
    logger.info(f"SARIMAX holdout — MAE: {mae:.4f}M  RMSE: {rmse:.4f}M  MAPE: {mape:.2%}  MSE(bias): {mse:+.4f}M")
    return mae, rmse, mape, mse


def evaluate_xgboost(client: MlflowClient) -> tuple[float, float, float]:
    df = pd.read_parquet(GOLD_ML_LOCAL_PATH)
    df.index = pd.to_datetime(df.index)
    feature_cols = [c for c in df.columns if c != "daily_ridership"]

    X_test = df[feature_cols].iloc[-TEST_DAYS:]
    y_test = df["daily_ridership"].iloc[-TEST_DAYS:] / 1_000_000

    model_uri = f"models:/{XGBOOST_MODEL_NAME}/Production"
    model = mlflow.xgboost.load_model(model_uri)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = mean_absolute_percentage_error(y_test, y_pred)
    mse = float(np.mean(np.array(y_pred) - np.array(y_test)))
    logger.info(f"XGBoost holdout — MAE: {mae:.4f}M  RMSE: {rmse:.4f}M  MAPE: {mape:.2%}  MSE(bias): {mse:+.4f}M")
    return mae, rmse, mape, mse


def run() -> str:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    logger.info("Evaluating SARIMAX champion")
    sarimax_mae, sarimax_rmse, sarimax_mape, sarimax_bias = evaluate_sarimax(client)

    logger.info("Evaluating XGBoost champion")
    xgb_mae, xgb_rmse, xgb_mape, xgb_bias = evaluate_xgboost(client)

    winner = SARIMAX_MODEL_NAME if sarimax_mae <= xgb_mae else XGBOOST_MODEL_NAME
    loser = XGBOOST_MODEL_NAME if winner == SARIMAX_MODEL_NAME else SARIMAX_MODEL_NAME

    logger.info(f"Champion: {winner} (MAE {min(sarimax_mae, xgb_mae):.4f}M)")

    with mlflow.start_run(run_name="model_comparison"):
        mlflow.log_metrics({
            "sarimax_mae": sarimax_mae, "sarimax_rmse": sarimax_rmse,
            "sarimax_mape": sarimax_mape, "sarimax_bias": sarimax_bias,
            "xgboost_mae": xgb_mae, "xgboost_rmse": xgb_rmse,
            "xgboost_mape": xgb_mape, "xgboost_bias": xgb_bias,
        })
        mlflow.log_param("champion_model", winner)
        mlflow.log_param("evaluation_date", str(date.today()))

    # Keep Production stage for winner; move loser to Staging
    for model_name, target_stage in [(winner, "Production"), (loser, "Staging")]:
        versions = client.search_model_versions(f"name='{model_name}'")
        if versions:
            latest = max(int(v.version) for v in versions)
            client.transition_model_version_stage(
                name=model_name, version=latest, stage=target_stage,
                archive_existing_versions=True,
            )
            logger.info(f"{model_name} v{latest} → {target_stage}")

    return winner


if __name__ == "__main__":
    run()
