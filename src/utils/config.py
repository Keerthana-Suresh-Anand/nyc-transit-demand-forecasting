import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(dotenv_path=BASE_DIR / ".env")

# AWS
AWS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET = os.getenv("AWS_SECRET_KEY")
AWS_REGION = os.getenv("AWS_REGION")
BUCKET = os.getenv("AWS_BUCKET_NAME")

# External APIs
NY_APP_TOKEN = os.getenv("NY_APP_TOKEN")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
TICKETMASTER_API_KEY = os.getenv("TICKETMASTER_API_KEY")

# MTA API
MTA_DATASET_ID = "5wq4-mkjj"            # 2025–present
MTA_HISTORICAL_DATASET_ID = "wujg-7c2s" # 2020–2024
MTA_HISTORICAL_END_DATE = "2024-12-31"
MTA_BASE_URL = f"https://data.ny.gov/resource/{MTA_DATASET_ID}.csv"
MTA_START_DATE = "2022-01-01"
MTA_LAG_DAYS = 6

# Weather API
WEATHER_LOCATION = "40.7812,-73.9665"  # Central Park, NYC
WEATHER_BASE_URL = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/"
WEATHER_FORECAST_DAYS = 14

# S3 keys — bronze layer
S3_MTA_PREFIX = "bronze/mta/"
S3_WEATHER_HIST_PREFIX = "bronze/weather/historical/"
S3_WEATHER_FORECAST_PREFIX = "bronze/weather/forecast/"
S3_EVENTS_PREFIX = "bronze/events/"
S3_MTA_WATERMARK = "bronze/mta/last_fetched.txt"
S3_WEATHER_WATERMARK = "bronze/weather/last_fetched.txt"

# S3 keys — silver layer
S3_SILVER_KEY = "silver/mta_weather_merged.parquet"

# S3 keys — gold layer
S3_GOLD_SARIMA_KEY = "gold/mta_sarima.parquet"
S3_GOLD_ML_KEY = "gold/mta_ml.parquet"
S3_FORECAST_PREFIX = "gold/forecasts/"
S3_LATEST_FORECAST_KEY = "gold/forecasts/latest_forecast.json"

# S3 keys — monitoring
S3_MONITORING_PREFIX = "monitoring/"
S3_PIPELINE_RUNS_PREFIX = "monitoring/pipeline_runs/"
S3_DRIFT_REPORT_PREFIX = "monitoring/reports/"
S3_RETRAIN_FLAG_KEY = "monitoring/retrain_flag.json"

# Local paths
DATA_DIR = BASE_DIR / "data"
SILVER_LOCAL_PATH = DATA_DIR / "silver" / "mta_weather_merged.parquet"
GOLD_SARIMA_LOCAL_PATH = DATA_DIR / "gold" / "mta_sarima.parquet"
GOLD_ML_LOCAL_PATH = DATA_DIR / "gold" / "mta_ml.parquet"
REPORTS_DIR = BASE_DIR / "reports" / "plots"

# MLflow
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{BASE_DIR}/mlflow.db")
MLFLOW_EXPERIMENT_NAME = "nyc_transit_forecasting"
SARIMAX_MODEL_NAME = "sarimax_production"
XGBOOST_MODEL_NAME = "xgboost_production"

# S3 keys — reports
S3_REPORTS_PREFIX = "reports/"
S3_SHAP_KEY = "reports/xgboost_shap_summary.png"
S3_SARIMAX_COEF_KEY = "reports/sarimax_coefficients.json"

# Ensemble weights
# Equal weights: 14-day rolling-origin walk-forward showed SARIMAX and XGBoost
# are statistically indistinguishable (bootstrapped 95% CIs on the MAE
# differences all span zero), so neither earns a heavier weight.
ENSEMBLE_SARIMAX_WEIGHT = 0.5
ENSEMBLE_XGB_WEIGHT = 0.5

# Training baseline written by evaluate_models, read by monitoring
S3_TRAINING_BASELINE_KEY = "monitoring/training_baseline.json"

# MLflow S3 persistence (used by GitHub Actions to sync tracking db and artifacts)
S3_MLFLOW_DB_KEY = "mlflow/mlflow.db"

# Monitoring thresholds
PSI_MODERATE_THRESHOLD = 0.1
PSI_CRITICAL_THRESHOLD = 0.25
MAE_RETRAIN_MULTIPLIER = 1.5  # retrain if rolling MAE > 1.5x training MAE
