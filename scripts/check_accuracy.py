import io
from datetime import date

import boto3
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from src.utils.config import AWS_KEY, AWS_REGION, AWS_SECRET, BUCKET, S3_FORECAST_PREFIX, S3_GOLD_SARIMA_KEY


def _s3():
    return boto3.client("s3", aws_access_key_id=AWS_KEY, aws_secret_access_key=AWS_SECRET, region_name=AWS_REGION)


def _get_parquet(s3, key):
    obj = s3.get_object(Bucket=BUCKET, Key=key)
    return pd.read_parquet(io.BytesIO(obj["Body"].read()))


def _list_keys(s3, prefix, suffix=""):
    paginator = s3.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            if not suffix or obj["Key"].endswith(suffix):
                keys.append(obj["Key"])
    return sorted(keys)


s3 = _s3()

gold = _get_parquet(s3, S3_GOLD_SARIMA_KEY)
gold.index = pd.to_datetime(gold.index)
gold = gold.sort_index()
print(f"Gold data: {gold.index.min().date()} to {gold.index.max().date()} ({len(gold)} rows)")

parquet_keys = _list_keys(s3, S3_FORECAST_PREFIX, ".parquet")
print(f"\nForecast files found: {len(parquet_keys)}")
for k in parquet_keys[-8:]:
    print(f"  {k}")

today = date.today()
rows = []
for key in parquet_keys[-8:]:
    df_fc = _get_parquet(s3, key)
    df_fc["date"] = pd.to_datetime(df_fc["date"]).dt.date
    past = df_fc[df_fc["date"] < today]
    print(f"\n{key.split('/')[-1]}: {len(df_fc)} rows, {past['date'].min()} to {past['date'].max()} in the past")
    for _, row in past.iterrows():
        ts = pd.Timestamp(row["date"])
        if ts in gold.index:
            actual_M = gold.loc[ts, "daily_ridership"] / 1_000_000
            err = row["ensemble_forecast_M"] - actual_M
            rows.append({
                "date": row["date"],
                "actual_M": round(actual_M, 3),
                "forecast_M": round(row["ensemble_forecast_M"], 3),
                "error_M": round(err, 3),
                "abs_pct_error": round(abs(err) / actual_M * 100, 1),
            })

if not rows:
    print("\nNo matched dates between forecasts and gold data.")
else:
    df = pd.DataFrame(rows).drop_duplicates("date").sort_values("date", ascending=False).reset_index(drop=True)
    print("\n" + df.to_string(index=False))
    print(f"\nMean absolute % error: {df['abs_pct_error'].mean():.1f}%")
    print(f"Mean absolute error:   {df['error_M'].abs().mean():.3f}M riders")
