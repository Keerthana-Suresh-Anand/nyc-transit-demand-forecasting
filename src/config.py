import os
from pathlib import Path
from dotenv import load_dotenv

# Find Project Root
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# S3 Configuration
BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")

# S3 file paths
RAW_MTA_KEY = "bronze/mta/mta_hourly_ridership_2026-02-05.csv"
RAW_WEATHER_KEY = "bronze/weather/nyc_weather_2025_2026_combined.csv"

# Local file paths
LOCAL_DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = LOCAL_DATA_DIR / "01_raw"
PROCESSED_DATA_DIR = LOCAL_DATA_DIR / "02_processed"

# Ensure local directories exist. Only create if it doesn't exist as a directory
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR]:
    if not directory.is_dir():
        # If a file exists with the same name, remove it first
        if directory.exists():
            directory.unlink()
        directory.mkdir(parents=True, exist_ok=True)

# File names
SAMPLE_MTA_PATH = RAW_DATA_DIR / "sample_raw_mta.csv"
SAMPLE_WEATHER_PATH = RAW_DATA_DIR / "sample_raw_weather.csv"

# This is where your Silver output will go locally
# SILVER_LOCAL_PATH = PROCESSED_DATA_DIR / "master_ridership_weather.csv"
