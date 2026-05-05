import io
import json
from datetime import date

import boto3
import pandas as pd
import streamlit as st

from src.utils.config import (
    AWS_KEY,
    AWS_REGION,
    AWS_SECRET,
    BUCKET,
    GOLD_SARIMA_LOCAL_PATH,
    S3_DRIFT_REPORT_PREFIX,
    S3_EVENTS_PREFIX,
    S3_FORECAST_PREFIX,
    S3_GOLD_SARIMA_KEY,
    S3_LATEST_FORECAST_KEY,
    S3_MTA_WATERMARK,
    S3_PIPELINE_RUNS_PREFIX,
    S3_SHAP_KEY,
    S3_WEATHER_FORECAST_PREFIX,
)


def _s3():
    return boto3.client(
        "s3",
        aws_access_key_id=AWS_KEY,
        aws_secret_access_key=AWS_SECRET,
        region_name=AWS_REGION,
    )


def _get_json(s3, key: str) -> dict | list | None:
    try:
        obj = s3.get_object(Bucket=BUCKET, Key=key)
        return json.loads(obj["Body"].read().decode("utf-8"))
    except Exception:
        return None


def _get_parquet(s3, key: str) -> pd.DataFrame | None:
    try:
        obj = s3.get_object(Bucket=BUCKET, Key=key)
        return pd.read_parquet(io.BytesIO(obj["Body"].read()))
    except Exception:
        return None


def _list_keys(s3, prefix: str, suffix: str = "") -> list[str]:
    try:
        paginator = s3.get_paginator("list_objects_v2")
        keys = []
        for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
            for obj in page.get("Contents", []):
                if not suffix or obj["Key"].endswith(suffix):
                    keys.append(obj["Key"])
        return sorted(keys)
    except Exception:
        return []


@st.cache_data(ttl=3600)
def load_forecast() -> dict | None:
    """Load latest_forecast.json from S3."""
    return _get_json(_s3(), S3_LATEST_FORECAST_KEY)


@st.cache_data(ttl=3600)
def load_history(days: int = 120) -> pd.DataFrame | None:
    """
    Load the last `days` rows of the gold SARIMA parquet.
    Columns: daily_ridership, temp, precip, snow, is_holiday, snow_lag1
    Falls back to local file if S3 is unavailable.
    """
    s3 = _s3()
    df = _get_parquet(s3, S3_GOLD_SARIMA_KEY)
    if df is None and GOLD_SARIMA_LOCAL_PATH.exists():
        df = pd.read_parquet(GOLD_SARIMA_LOCAL_PATH)
    if df is None:
        return None
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df.iloc[-days:] if len(df) > days else df


@st.cache_data(ttl=3600)
def load_weather_forecast() -> pd.DataFrame | None:
    """Load the most recent 14-day weather forecast CSV from S3 bronze layer."""
    s3 = _s3()
    keys = _list_keys(s3, S3_WEATHER_FORECAST_PREFIX, ".csv")
    if not keys:
        return None
    try:
        obj = s3.get_object(Bucket=BUCKET, Key=keys[-1])
        df = pd.read_csv(io.StringIO(obj["Body"].read().decode("utf-8")))
        df["datetime"] = pd.to_datetime(df["datetime"])
        return df.sort_values("datetime").reset_index(drop=True)
    except Exception:
        return None


@st.cache_data(ttl=3600)
def load_drift_report() -> dict | None:
    """Load the most recent drift report JSON from S3."""
    s3 = _s3()
    keys = _list_keys(s3, S3_DRIFT_REPORT_PREFIX, ".json")
    return _get_json(s3, keys[-1]) if keys else None


@st.cache_data(ttl=3600)
def load_shap_image() -> bytes | None:
    """Load XGBoost SHAP summary PNG bytes from S3."""
    try:
        s3 = _s3()
        obj = s3.get_object(Bucket=BUCKET, Key=S3_SHAP_KEY)
        return obj["Body"].read()
    except Exception:
        return None


@st.cache_data(ttl=3600)
def load_past_forecasts_vs_actuals() -> pd.DataFrame | None:
    """
    Load the last 8 weekly forecast parquets from S3, cross-join predicted dates
    that are now in the past with actuals from the gold parquet.
    Returns columns: forecast_run_date, date, ensemble_forecast_M,
                     sarimax_forecast_M, actual_M, error_M, abs_pct_error, week.
    """
    s3 = _s3()
    history = load_history(days=180)
    if history is None:
        return None

    parquet_keys = _list_keys(s3, S3_FORECAST_PREFIX, ".parquet")
    if not parquet_keys:
        return None

    today = date.today()
    rows = []
    for key in parquet_keys[-8:]:
        run_date_str = key.split("forecast_")[-1].replace(".parquet", "")
        df_fc = _get_parquet(s3, key)
        if df_fc is None:
            continue
        df_fc["date"] = pd.to_datetime(df_fc["date"]).dt.date
        for _, row in df_fc[df_fc["date"] < today].iterrows():
            ts = pd.Timestamp(row["date"])
            if ts in history.index:
                actual_M = history.loc[ts, "daily_ridership"] / 1_000_000
                err = row["ensemble_forecast_M"] - actual_M
                rows.append({
                    "forecast_run_date": run_date_str,
                    "date": row["date"],
                    "ensemble_forecast_M": row["ensemble_forecast_M"],
                    "sarimax_forecast_M": row["sarimax_forecast_M"],
                    "actual_M": actual_M,
                    "error_M": err,
                    "abs_pct_error": abs(err) / actual_M * 100,
                })

    if not rows:
        return None
    df = pd.DataFrame(rows).sort_values("date", ascending=False).reset_index(drop=True)
    df["week"] = pd.to_datetime(df["date"]).dt.to_period("W")
    return df


@st.cache_data(ttl=3600)
def load_pipeline_status() -> dict:
    """Aggregate pipeline run dates and health from S3 watermarks and run logs."""
    s3 = _s3()
    status: dict = {
        "last_ingestion_date": None,
        "last_training_date": None,
        "last_forecast_date": None,
        "ingestion_status": None,
        "training_status": None,
        "prediction_status": None,
        "training_row_count": None,
    }

    try:
        obj = s3.get_object(Bucket=BUCKET, Key=S3_MTA_WATERMARK)
        status["last_ingestion_date"] = obj["Body"].read().decode().strip()
    except Exception:
        pass

    all_run_keys = _list_keys(s3, S3_PIPELINE_RUNS_PREFIX, ".json")
    for pipeline_type in ("training", "ingestion", "prediction"):
        keys = [k for k in all_run_keys if f"/{pipeline_type}_" in k]
        if not keys:
            continue
        run = _get_json(s3, keys[-1])
        if not run:
            continue
        if pipeline_type == "training":
            status["last_training_date"] = run.get("run_date")
            status["training_status"] = run.get("status")
        elif pipeline_type == "prediction":
            status["last_forecast_date"] = run.get("run_date")
            status["prediction_status"] = run.get("status")
        elif pipeline_type == "ingestion":
            status["ingestion_status"] = run.get("status")

    if not status["last_forecast_date"]:
        fc = load_forecast()
        if fc:
            status["last_forecast_date"] = fc.get("run_date")

    history = load_history(days=5)
    if history is not None:
        status["training_row_count"] = len(history)

    return status


@st.cache_data(ttl=3600)
def load_events() -> list[dict]:
    """Load NYC event records from S3 bronze layer (last 4 weekly files)."""
    s3 = _s3()
    keys = _list_keys(s3, S3_EVENTS_PREFIX, ".json")
    events: list[dict] = []
    for key in keys[-4:]:
        data = _get_json(s3, key)
        if isinstance(data, list):
            events.extend(data)
    return events
