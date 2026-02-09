import os
import requests
import boto3
import io
import time
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime

# 1. Setup
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=BASE_DIR / ".env")

# 2. Config
API_KEY = os.getenv("WEATHER_API_KEY")
BUCKET = os.getenv("AWS_BUCKET_NAME")
S3_CLIENT = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY"),
    aws_secret_access_key=os.getenv("AWS_SECRET_KEY"),
    region_name=os.getenv("AWS_REGION"),
)


def get_monthly_ranges(start_year, start_month):
    """Generates a list of (start_date, end_date) tuples from start to today."""
    ranges = []
    current_date = datetime(start_year, start_month, 1)
    today = datetime.now()

    while current_date < today:
        # Start of month
        month_start = current_date.strftime("%Y-%m-%d")

        # End of month (handling year wrap-around)
        if current_date.month == 12:
            next_month = datetime(current_date.year + 1, 1, 1)
        else:
            next_month = datetime(current_date.year, current_date.month + 1, 1)

        # If next_month is in the future, use today
        actual_end = min(next_month, today)
        month_end = actual_end.strftime("%Y-%m-%d")

        ranges.append((month_start, month_end))
        current_date = next_month
    return ranges


def run_weather_upload():
    location = "40.7812,-73.9665"  # NYC Central Park
    base_url = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/"

    # Generate chunks for Jan 2025 through today
    chunks = get_monthly_ranges(2025, 1)
    all_data_frames = []

    print(f"🚀 Starting monthly weather fetch for {len(chunks)} months...")

    for start, end in chunks:
        print(f"🌦️  Fetching: {start} to {end}...")
        url = (
            f"{base_url}{location}/{start}/{end}?"
            f"unitGroup=metric&contentType=csv&include=days&key={API_KEY}"
        )

        try:
            response = requests.get(url, timeout=60)

            if response.status_code == 429:
                print(
                    "❌ Rate limited again! You must wait the full hour before trying this script."
                )
                return

            response.raise_for_status()
            df = pd.read_csv(io.StringIO(response.text))
            all_data_frames.append(df)

            time.sleep(2)

        except Exception as e:
            print(f"❌ Error at {start}: {e}")
            break

    # 3. Combine and Upload to S3
    if all_data_frames:
        final_df = pd.concat(all_data_frames, ignore_index=True)
        csv_buffer = io.StringIO()
        final_df.to_csv(csv_buffer, index=False)

        s3_key = "bronze/nyc_weather_daily_2025_2026.csv"
        S3_CLIENT.put_object(Bucket=BUCKET, Key=s3_key, Body=csv_buffer.getvalue())
        print(f"✅ Success! Saved {len(final_df)} days of weather to S3.")


if __name__ == "__main__":
    run_weather_upload()
