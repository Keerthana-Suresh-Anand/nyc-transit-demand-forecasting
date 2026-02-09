import pandas as pd
import boto3
import os
from src.config import (
    BUCKET_NAME,
    RAW_MTA_KEY,
    RAW_WEATHER_KEY,
    LOCAL_DATA_DIR,
    SAMPLE_MTA_PATH,
    SAMPLE_WEATHER_PATH,
)


def create_local_samples():
    LOCAL_DATA_DIR.mkdir(exist_ok=True)

    s3 = boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY"),
        aws_secret_access_key=os.getenv("AWS_SECRET_KEY"),
    )

    # 1. Sample MTA data
    print(f"Sampling MTA data from: {RAW_MTA_KEY}")
    try:
        mta_obj = s3.get_object(Bucket=BUCKET_NAME, Key=RAW_MTA_KEY)
        mta_sample = pd.read_csv(mta_obj["Body"], nrows=5000).sample(n=1000)
        mta_sample.to_csv(SAMPLE_MTA_PATH, index=False)
        print("MTA Sample created.")
    except Exception as e:
        print(f"Error fetching MTA data: {e}")

    # 2. Sample Weather data
    print(f"Sampling Weather data from: {RAW_WEATHER_KEY}")
    try:
        weather_obj = s3.get_object(Bucket=BUCKET_NAME, Key=RAW_WEATHER_KEY)
        weather_df = pd.read_csv(weather_obj["Body"])
        weather_df.head(30).to_csv(SAMPLE_WEATHER_PATH, index=False)
        print("Weather Sample created.")
    except Exception as e:
        print(f"Error fetching Weather data: {e}")


if __name__ == "__main__":
    create_local_samples()
