"""
Generates a 14-day ensemble forecast using the Production SARIMAX and XGBoost models.
Writes forecast to S3 as both a timestamped parquet and latest_forecast.json.
"""
import pickle
import warnings
from datetime import date, timedelta
from pathlib import Path

import mlflow
import mlflow.statsmodels
import mlflow.xgboost
import numpy as np
import pandas as pd
import yaml
from mlflow import MlflowClient
from sklearn.preprocessing import MinMaxScaler

from src.utils.config import (
    ENSEMBLE_SARIMAX_WEIGHT,
    ENSEMBLE_XGB_WEIGHT,
    GOLD_ML_LOCAL_PATH,
    GOLD_SARIMA_LOCAL_PATH,
    MLFLOW_TRACKING_URI,
    PIPELINE_IMAGE_DIGEST,
    S3_FORECAST_PREFIX,
    S3_LATEST_FORECAST_KEY,
    S3_WEATHER_FORECAST_PREFIX,
    SARIMAX_EXOG_COLS,
    SARIMAX_MODEL_NAME,
    XGBOOST_MODEL_NAME,
)
from src.utils.features import cast_categoricals
from src.utils.logger import get_logger
from src.utils.s3_helpers import (
    get_s3_client,
    list_s3_files,
    read_s3_csv,
    write_s3_json,
    write_s3_parquet,
)

warnings.filterwarnings("ignore")
logger = get_logger(__name__)

ML_FEATURE_COLS = None  # resolved at runtime from training data
FORECAST_DAYS = 14


def _load_production_scaler() -> MinMaxScaler | None:
    """Load the scaler persisted during the Production SARIMAX training run.
    Returns None if not found (e.g. model trained before scaler persistence was added).
    """
    try:
        client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        versions = client.search_model_versions(f"name='{SARIMAX_MODEL_NAME}'")
        prod = next((v for v in versions if v.current_stage == "Production"), None)
        if prod is None:
            return None
        local_path = mlflow.artifacts.download_artifacts(
            run_id=prod.run_id,
            artifact_path="sarimax_scaler.pkl",
            tracking_uri=MLFLOW_TRACKING_URI,
        )
        with open(local_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        logger.warning(f"Could not load Production scaler from MLflow: {e}")
        return None


def _production_versions() -> dict:
    """Resolve the Production version + run_id of each model so the forecast can be
    traced back to the exact registered artifacts that produced it.
    """
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    out: dict = {}
    for label, name in (("sarimax", SARIMAX_MODEL_NAME), ("xgboost", XGBOOST_MODEL_NAME)):
        try:
            prod = [v for v in client.search_model_versions(f"name='{name}'")
                    if v.current_stage == "Production"]
            if prod:
                v = max(prod, key=lambda x: int(x.version))
                out[label] = {"version": int(v.version), "run_id": v.run_id}
            else:
                out[label] = None
        except Exception as e:
            logger.warning(f"Could not resolve Production version for {name}: {e}")
            out[label] = None
    return out


def _gold_dvc_md5() -> str | None:
    """md5 of the gold SARIMA parquet from its DVC pointer, tying the forecast to a
    data version. Best-effort — returns None if the pointer isn't present.
    """
    try:
        with open(Path("data/gold/mta_sarima.parquet.dvc")) as f:
            return yaml.safe_load(f)["outs"][0]["md5"]
    except Exception:
        return None


def load_latest_weather_forecast(s3) -> pd.DataFrame:
    keys = sorted(list_s3_files(s3, S3_WEATHER_FORECAST_PREFIX))
    if not keys:
        raise FileNotFoundError("No weather forecast files found in S3.")
    df = read_s3_csv(s3, keys[-1])
    df["datetime"] = pd.to_datetime(df["datetime"]).dt.date
    return df


def sarimax_forecast(df_sarima: pd.DataFrame, weather_fcst: pd.DataFrame, start_date: date) -> pd.Series:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model = mlflow.statsmodels.load_model(f"models:/{SARIMAX_MODEL_NAME}/Production")

    scaler = _load_production_scaler()
    if scaler is None:
        logger.warning("Scaler not in MLflow — re-fitting on all available data (next training run will fix this)")
        scaler = MinMaxScaler()
        scaler.fit(df_sarima[SARIMAX_EXOG_COLS])

    import holidays as hol
    us_holidays = hol.US(years=[start_date.year, start_date.year + 1])

    future_dates = [start_date + timedelta(days=i) for i in range(FORECAST_DAYS)]
    weather_map = dict(zip(weather_fcst["datetime"], weather_fcst.itertuples()))

    rows = []
    last_snow = float(df_sarima["snow"].iloc[-1])
    for d in future_dates:
        row_weather = weather_map.get(d)
        temp = row_weather.temp if row_weather and hasattr(row_weather, "temp") else df_sarima["temp"].mean()
        precip = row_weather.precip if row_weather and hasattr(row_weather, "precip") else 0.0
        snow = row_weather.snow if row_weather and hasattr(row_weather, "snow") else 0.0
        rows.append({"temp": temp, "precip": precip, "snow_lag1": last_snow,
                     "is_holiday": int(d in us_holidays)})
        last_snow = snow

    future_exog = pd.DataFrame(rows, index=pd.to_datetime(future_dates))
    future_exog_scaled = pd.DataFrame(
        scaler.transform(future_exog), index=future_exog.index, columns=SARIMAX_EXOG_COLS
    )

    forecast = model.get_forecast(steps=FORECAST_DAYS, exog=future_exog_scaled)
    return forecast.predicted_mean, forecast.conf_int()


def xgboost_forecast(df_ml: pd.DataFrame, weather_fcst: pd.DataFrame, start_date: date) -> np.ndarray:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model = mlflow.xgboost.load_model(f"models:/{XGBOOST_MODEL_NAME}/Production")

    feature_cols = [c for c in df_ml.columns if c != "daily_ridership"]
    df_rolling = df_ml.copy()

    import holidays as hol
    us_holidays = hol.US(years=[start_date.year, start_date.year + 1])
    weather_map = dict(zip(weather_fcst["datetime"], weather_fcst.itertuples()))

    predictions = []
    for step in range(FORECAST_DAYS):
        pred_date = start_date + timedelta(days=step)
        last_row = df_rolling.iloc[-1].copy()

        # Build the next feature row using known future values + predicted ridership lags
        rw = weather_map.get(pred_date)
        next_row = {
            "day_of_week": pd.Timestamp(pred_date).dayofweek,
            "month": pred_date.month,
            "is_weekend": int(pd.Timestamp(pred_date).dayofweek >= 5),
            "is_holiday": int(pred_date in us_holidays),
            "temp": rw.temp if rw and hasattr(rw, "temp") else last_row.get("temp_lag1", df_ml["temp_lag1"].mean()),
            "precip": rw.precip if rw and hasattr(rw, "precip") else 0.0,
            "snow": rw.snow if rw and hasattr(rw, "snow") else 0.0,
            "snow_lag1": last_row.get("snow", 0.0),
            "temp_lag1": last_row.get("temp", df_ml["temp_lag1"].mean()),
            "precip_lag1": last_row.get("precip", 0.0),
        }
        # Ridership lags — use predicted values for lags already in forecast window
        for lag in [1, 2, 3, 7, 14]:
            lag_idx = -(lag)
            if len(predictions) >= lag:
                next_row[f"ridership_lag{lag}"] = predictions[-lag]
            else:
                next_row[f"ridership_lag{lag}"] = df_rolling["daily_ridership"].iloc[lag_idx] / 1_000_000

        # Rolling stats — approximate using last available window
        recent_ridership = list(df_rolling["daily_ridership"].iloc[-14:] / 1_000_000) + predictions
        next_row["ridership_14d_avg"] = np.mean(recent_ridership[-14:])
        next_row["ridership_7d_std"] = np.std(recent_ridership[-7:])

        X_next = cast_categoricals(pd.DataFrame([next_row])[feature_cols])
        pred = float(model.predict(X_next)[0])
        predictions.append(pred)

    return np.array(predictions)


def run() -> None:
    logger.info("Starting forecast generation")
    s3 = get_s3_client()

    df_sarima = pd.read_parquet(GOLD_SARIMA_LOCAL_PATH)
    df_sarima.index = pd.to_datetime(df_sarima.index)
    df_ml = pd.read_parquet(GOLD_ML_LOCAL_PATH)
    df_ml.index = pd.to_datetime(df_ml.index)

    weather_fcst = load_latest_weather_forecast(s3)
    logger.info(f"Weather forecast loaded: {len(weather_fcst)} days")

    # Start from the day after the last actual observation, not today, so there
    # is no gap between the actuals line and the forecast line on the dashboard.
    start_date = (df_sarima.index.max() + timedelta(days=1)).date()
    logger.info(f"Forecast start date: {start_date}")

    logger.info("Running SARIMAX forecast")
    sarimax_pred, conf_int = sarimax_forecast(df_sarima, weather_fcst, start_date)

    logger.info("Running XGBoost forecast")
    xgb_pred = xgboost_forecast(df_ml, weather_fcst, start_date)

    future_dates = pd.date_range(start=start_date, periods=FORECAST_DAYS)
    ensemble_pred = ENSEMBLE_SARIMAX_WEIGHT * sarimax_pred.values + ENSEMBLE_XGB_WEIGHT * xgb_pred

    df_forecast = pd.DataFrame({
        "date": future_dates,
        "sarimax_forecast_M": sarimax_pred.values,
        "xgboost_forecast_M": xgb_pred,
        "ensemble_forecast_M": ensemble_pred,
        "ci_lower": conf_int.iloc[:, 0].values,
        "ci_upper": conf_int.iloc[:, 1].values,
    })

    run_date = str(date.today())
    parquet_key = f"{S3_FORECAST_PREFIX}forecast_{run_date}.parquet"
    write_s3_parquet(s3, df_forecast, parquet_key)

    forecast_json = {
        "run_date": run_date,
        "forecast_horizon_days": FORECAST_DAYS,
        "sarimax_weight": ENSEMBLE_SARIMAX_WEIGHT,
        "xgboost_weight": ENSEMBLE_XGB_WEIGHT,
        "model_versions": _production_versions(),
        "gold_sarima_dvc_md5": _gold_dvc_md5(),
        "image_digest": PIPELINE_IMAGE_DIGEST,
        "last_actual_date": str(df_sarima.index.max().date()),
        "forecasts": df_forecast.to_dict(orient="records"),
    }
    write_s3_json(s3, forecast_json, S3_LATEST_FORECAST_KEY)

    logger.info(f"Forecast complete: {FORECAST_DAYS} days written to S3")


if __name__ == "__main__":
    run()
