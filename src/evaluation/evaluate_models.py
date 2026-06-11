"""
Compares each newly trained model version against the current Production version.
Promotes to Production only if the new version has lower MAE on the holdout set.
Both models go to Production (ensemble uses both); winner between families is metadata only.
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
    GOLD_ML_LOCAL_PATH,
    GOLD_SARIMA_LOCAL_PATH,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    SARIMAX_MODEL_NAME,
    XGBOOST_MODEL_NAME,
)
from src.utils.logger import get_logger

warnings.filterwarnings("ignore")
logger = get_logger(__name__)

TEST_DAYS = 30
SARIMAX_EXOG_COLS = ["temp", "precip", "snow_lag1", "is_holiday"]


def evaluate_sarimax(model_uri: str) -> tuple[float, float, float, float]:
    df = pd.read_parquet(GOLD_SARIMA_LOCAL_PATH)
    df.index = pd.to_datetime(df.index)
    df = df.asfreq("D")
    y = df["daily_ridership"] / 1_000_000

    train_idx = y.iloc[:-TEST_DAYS].index
    test_idx = y.iloc[-TEST_DAYS:].index
    test_y = y.loc[test_idx]

    scaler = MinMaxScaler()
    scaler.fit(df.loc[train_idx, SARIMAX_EXOG_COLS])
    test_exog = pd.DataFrame(
        scaler.transform(df.loc[test_idx, SARIMAX_EXOG_COLS]),
        index=test_idx, columns=SARIMAX_EXOG_COLS,
    )

    model = mlflow.statsmodels.load_model(model_uri)
    y_pred = model.get_forecast(steps=len(test_y), exog=test_exog).predicted_mean

    mae = mean_absolute_error(test_y, y_pred)
    rmse = np.sqrt(mean_squared_error(test_y, y_pred))
    mape = mean_absolute_percentage_error(test_y, y_pred)
    bias = float(np.mean(np.array(y_pred) - np.array(test_y)))
    return mae, rmse, mape, bias


def evaluate_xgboost(model_uri: str) -> tuple[float, float, float, float]:
    df = pd.read_parquet(GOLD_ML_LOCAL_PATH)
    df.index = pd.to_datetime(df.index)
    feature_cols = [c for c in df.columns if c != "daily_ridership"]

    X_test = df[feature_cols].iloc[-TEST_DAYS:]
    y_test = df["daily_ridership"].iloc[-TEST_DAYS:] / 1_000_000

    model = mlflow.xgboost.load_model(model_uri)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = mean_absolute_percentage_error(y_test, y_pred)
    bias = float(np.mean(np.array(y_pred) - np.array(y_test)))
    return mae, rmse, mape, bias


def _latest_version(client: MlflowClient, model_name: str) -> int:
    versions = client.search_model_versions(f"name='{model_name}'")
    return max(int(v.version) for v in versions)


def _production_version(client: MlflowClient, model_name: str) -> int | None:
    versions = client.search_model_versions(f"name='{model_name}'")
    prod = [v for v in versions if v.current_stage == "Production"]
    return max(int(v.version) for v in prod) if prod else None


def _promote(client: MlflowClient, model_name: str, version: int) -> None:
    client.transition_model_version_stage(
        name=model_name, version=version, stage="Production",
        archive_existing_versions=True,
    )


def _evaluate_and_gate(
    client: MlflowClient,
    model_name: str,
    evaluate_fn,
) -> tuple[float, float, float, float]:
    """Gate Production promotion: promote only if new version MAE < current Production MAE."""
    new_ver = _latest_version(client, model_name)
    prod_ver = _production_version(client, model_name)

    new_mae, new_rmse, new_mape, new_bias = evaluate_fn(f"models:/{model_name}/{new_ver}")
    logger.info(
        f"{model_name} v{new_ver} (candidate) — "
        f"MAE: {new_mae:.4f}M  RMSE: {new_rmse:.4f}M  MAPE: {new_mape:.2%}  bias: {new_bias:+.4f}M"
    )

    if prod_ver is None:
        _promote(client, model_name, new_ver)
        logger.info(f"{model_name} v{new_ver} → Production (no prior champion)")
    elif prod_ver == new_ver:
        logger.info(f"{model_name} v{new_ver} already in Production — skipping gate")
    else:
        old_mae, _, _, _ = evaluate_fn(f"models:/{model_name}/{prod_ver}")
        logger.info(f"{model_name} v{prod_ver} (Production) — MAE: {old_mae:.4f}M")
        if new_mae < old_mae:
            _promote(client, model_name, new_ver)
            logger.info(
                f"{model_name} v{new_ver} → Production "
                f"(MAE {new_mae:.4f}M < {old_mae:.4f}M, improvement: {old_mae - new_mae:.4f}M)"
            )
        else:
            logger.info(
                f"{model_name} v{new_ver} NOT promoted "
                f"(MAE {new_mae:.4f}M >= {old_mae:.4f}M) — keeping v{prod_ver} in Production"
            )

    return new_mae, new_rmse, new_mape, new_bias


def run() -> str:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    logger.info("Evaluating SARIMAX")
    sarimax_mae, sarimax_rmse, sarimax_mape, sarimax_bias = _evaluate_and_gate(
        client, SARIMAX_MODEL_NAME, evaluate_sarimax,
    )

    logger.info("Evaluating XGBoost")
    xgb_mae, xgb_rmse, xgb_mape, xgb_bias = _evaluate_and_gate(
        client, XGBOOST_MODEL_NAME, evaluate_xgboost,
    )

    winner = SARIMAX_MODEL_NAME if sarimax_mae <= xgb_mae else XGBOOST_MODEL_NAME
    logger.info(f"Champion model family: {winner} (MAE {min(sarimax_mae, xgb_mae):.4f}M)")

    with mlflow.start_run(run_name="model_comparison"):
        mlflow.log_metrics({
            "sarimax_mae": sarimax_mae, "sarimax_rmse": sarimax_rmse,
            "sarimax_mape": sarimax_mape, "sarimax_bias": sarimax_bias,
            "xgboost_mae": xgb_mae, "xgboost_rmse": xgb_rmse,
            "xgboost_mape": xgb_mape, "xgboost_bias": xgb_bias,
        })
        mlflow.log_param("champion_model", winner)
        mlflow.log_param("evaluation_date", str(date.today()))

    return winner


if __name__ == "__main__":
    run()
