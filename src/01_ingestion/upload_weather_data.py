import os
import requests
import boto3
import io
import time
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime

# 1. Setup Paths & Load Environment Variables
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=BASE_DIR / ".env")

# 2. AWS & API Configuration
API_KEY = os.getenv("WEATHER_API_KEY")
BUCKET = os.getenv("AWS_BUCKET_NAME")
S3_CLIENT = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY"),
    aws_secret_access_key=os.getenv("AWS_SECRET_KEY"),
    region_name=os.getenv("AWS_REGION"),
)


def run_weather_upload():
    location = "40.7812,-73.9665"
    base_url = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/"

    # We create monthly chunks to be as gentle as possible to the API
    months = [
        ("2025-01-01", "2025-01-31"),
        ("2025-02-01", "2025-02-28"),
        ("2025-03-01", "2025-03-31"),
        ("2025-04-01", "2025-04-30"),
        ("2025-05-01", "2025-05-31"),
        ("2025-06-01", "2025-06-30"),
        ("2025-07-01", "2025-07-31"),
        ("2025-08-01", "2025-08-31"),
        ("2025-09-01", "2025-09-30"),
        ("2025-10-01", "2025-10-31"),
        ("2025-11-01", "2025-11-30"),
        ("2025-12-01", "2025-12-31"),
        ("2026-01-01", datetime.now().strftime("%Y-%m-%d")),
    ]

    all_data_frames = []

    for start, end in months:
        print(f"🌦️  Fetching: {start} to {end}...")
        url = (
            f"{base_url}{location}/{start}/{end}?"
            f"unitGroup=metric&contentType=csv&include=days&key={API_KEY}"
        )

        try:
            response = requests.get(url, timeout=60)

            if response.status_code == 429:
                print(f"⚠️  Hit a limit at {start}. Pausing for 10 seconds...")
                time.sleep(10)  # Longer pause if we hit a limit
                response = requests.get(url, timeout=60)  # Try one more time

            response.raise_for_status()
            df = pd.read_csv(io.StringIO(response.text))
            all_data_frames.append(df)
            time.sleep(2)

        except Exception as e:
            print(f"❌  Stopped at {start}: {e}")
            break  # Exit loop so we can at least save what we got

    if all_data_frames:
        final_df = pd.concat(all_data_frames, ignore_index=True)
        csv_buffer = io.StringIO()
        final_df.to_csv(csv_buffer, index=False)

        s3_key = "bronze/nyc_weather_2025_2026_combined.csv"
        S3_CLIENT.put_object(Bucket=BUCKET, Key=s3_key, Body=csv_buffer.getvalue())
        print(f"🚀  Success! Saved {len(final_df)} days of weather to S3.")


if __name__ == "__main__":
    run_weather_upload()
