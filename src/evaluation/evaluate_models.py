"""
Compares each newly trained model version against the current Production version.
Promotes to Production only if the new version has lower MAE on the holdout set.
Both models go to Production (ensemble uses both); winner between families is metadata only.
After evaluation, writes a training baseline JSON to S3 containing the ensemble MAE
so the monitoring pipeline has a meaningful threshold for triggering retrains.
"""
import warnings

import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from src.utils.config import (
    ENSEMBLE_SARIMAX_WEIGHT,
    ENSEMBLE_XGB_WEIGHT,
    GOLD_ML_LOCAL_PATH,
    GOLD_SARIMA_LOCAL_PATH,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    S3_TRAINING_BASELINE_KEY,
    SARIMAX_MODEL_NAME,
    XGBOOST_MODEL_NAME,
)
from src.utils.features import CATEGORICAL_FEATURES, cast_categoricals
from src.utils.logger import get_logger
from src.utils.s3_helpers import get_s3_client, write_s3_json

warnings.filterwarnings("ignore")
logger = get_logger(__name__)

TEST_DAYS = 60
SARIMAX_EXOG_COLS = ["temp", "precip", "snow_lag1", "is_holiday"]


def evaluate_sarimax(model_uri: str) -> tuple[float, float, float, float, np.ndarray, np.ndarray]:
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
    return mae, rmse, mape, bias, np.array(y_pred), np.array(test_y)


def _xgboost_iterative_predict(model, df: pd.DataFrame, test_start_idx: int, n_steps: int) -> np.ndarray:
    """Predict iteratively, propagating predicted ridership into lag features.

    Matches the inference loop in generate_forecast.py: calendar and weather
    features use actual values (known at forecast time), but ridership lags are
    filled with the model's own prior predictions rather than true values.
    """
    feature_cols = [c for c in df.columns if c != "daily_ridership"]
    ridership_lag_cols = {
        c: int(c.replace("ridership_lag", ""))
        for c in feature_cols if c.startswith("ridership_lag")
    }

    predictions: list[float] = []

    for step in range(n_steps):
        target_idx = test_start_idx + step
        target_row = df.iloc[target_idx]

        next_row: dict = {}
        for col in feature_cols:
            if col in ridership_lag_cols:
                lag = ridership_lag_cols[col]
                if len(predictions) >= lag:
                    next_row[col] = predictions[-lag]
                else:
                    next_row[col] = df["daily_ridership"].iloc[target_idx - lag] / 1_000_000
            elif col == "ridership_14d_avg":
                history = list(df["daily_ridership"].iloc[max(0, target_idx - 14):target_idx] / 1_000_000)
                window = (history + predictions)[-14:]
                next_row[col] = float(np.mean(window)) if window else 0.0
            elif col == "ridership_7d_std":
                history = list(df["daily_ridership"].iloc[max(0, target_idx - 14):target_idx] / 1_000_000)
                window = (history + predictions)[-7:]
                next_row[col] = float(np.std(window)) if len(window) >= 2 else 0.0
            elif col in CATEGORICAL_FEATURES:
                # Keep as int — floating then re-casting to a fixed integer
                # category range would turn the value into NaN.
                next_row[col] = int(target_row[col])
            else:
                next_row[col] = float(target_row[col])

        X_next = cast_categoricals(pd.DataFrame([next_row])[feature_cols])
        predictions.append(float(model.predict(X_next)[0]))

    return np.array(predictions)


def evaluate_xgboost(model_uri: str) -> tuple[float, float, float, float, np.ndarray, np.ndarray]:
    df = pd.read_parquet(GOLD_ML_LOCAL_PATH)
    df.index = pd.to_datetime(df.index)
    df = cast_categoricals(df)  # parquet does not preserve category dtype

    test_start_idx = len(df) - TEST_DAYS
    y_test = df["daily_ridership"].iloc[-TEST_DAYS:] / 1_000_000

    model = mlflow.xgboost.load_model(model_uri)
    y_pred = _xgboost_iterative_predict(model, df, test_start_idx, TEST_DAYS)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = mean_absolute_percentage_error(y_test, y_pred)
    bias = float(np.mean(y_pred - np.array(y_test)))
    return mae, rmse, mape, bias, y_pred, np.array(y_test)


def _promote(client: MlflowClient, model_name: str, version: int) -> None:
    client.transition_model_version_stage(
        name=model_name, version=version, stage="Production",
        archive_existing_versions=True,
    )


def _evaluate_and_gate(
    client: MlflowClient,
    model_name: str,
    evaluate_fn,
) -> tuple[float, float, float, float, np.ndarray, np.ndarray]:
    """Gate Production promotion: promote only if new version MAE < current Production MAE."""
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        logger.info(f"{model_name} — no registered versions found, skipping")
        return 0.0, 0.0, 0.0, 0.0, np.array([]), np.array([])

    new_ver = max(int(v.version) for v in versions)
    prod_versions = [v for v in versions if v.current_stage == "Production"]
    prod_ver = max(int(v.version) for v in prod_versions) if prod_versions else None

    new_mae, new_rmse, new_mape, new_bias, new_pred, new_actual = evaluate_fn(
        f"models:/{model_name}/{new_ver}"
    )
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
        old_mae, _, _, _, _, _ = evaluate_fn(f"models:/{model_name}/{prod_ver}")
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

    return new_mae, new_rmse, new_mape, new_bias, new_pred, new_actual


def run() -> str:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    logger.info("Evaluating SARIMAX")
    sarimax_mae, sarimax_rmse, sarimax_mape, sarimax_bias, sarimax_pred, y_actual = (
        _evaluate_and_gate(client, SARIMAX_MODEL_NAME, evaluate_sarimax)
    )

    logger.info("Evaluating XGBoost")
    xgb_mae, xgb_rmse, xgb_mape, xgb_bias, xgb_pred, _ = (
        _evaluate_and_gate(client, XGBOOST_MODEL_NAME, evaluate_xgboost)
    )

    winner = SARIMAX_MODEL_NAME if sarimax_mae <= xgb_mae else XGBOOST_MODEL_NAME
    logger.info(f"Champion model family: {winner} (MAE {min(sarimax_mae, xgb_mae):.4f}M)")

    if sarimax_pred.size > 0 and xgb_pred.size > 0:
        ensemble_pred = ENSEMBLE_SARIMAX_WEIGHT * sarimax_pred + ENSEMBLE_XGB_WEIGHT * xgb_pred
        ensemble_mae = float(np.mean(np.abs(ensemble_pred - y_actual)))
        logger.info(f"Ensemble holdout MAE: {ensemble_mae:.4f}M")
        s3 = get_s3_client()
        write_s3_json(s3, {
            "ensemble_mae": ensemble_mae,
            "sarimax_mae": sarimax_mae,
            "xgboost_mae": xgb_mae,
            "champion_model": winner,
        }, S3_TRAINING_BASELINE_KEY)
        logger.info("Training baseline written to S3")

    return winner


if __name__ == "__main__":
    run()
