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

# Docker image digest of the running pipeline container, injected by the GitHub
# Actions workflows (docker inspect after pull). Recorded in the pipeline run logs
# and latest_forecast.json so every output is traceable to the exact code image —
# alongside the data version (DVC md5) and model versions already recorded.
# None when running outside the containerized workflows (local runs).
PIPELINE_IMAGE_DIGEST = os.getenv("PIPELINE_IMAGE_DIGEST")

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

# Exogenous regressors for SARIMAX — single source of truth. Used by the trainer
# (fit + scaler) and the prediction pipeline (building future exog). Order matters:
# the persisted scaler is fit on these columns in this order.
SARIMAX_EXOG_COLS = ["temp", "precip", "snow_lag1", "is_holiday"]

# Holdout size (days) for model evaluation. Single source of truth shared by the
# training scripts (which compute + log the honest out-of-sample metrics) and the
# champion gate (which reads those logged metrics). Keep these in lockstep — a
# mismatch silently overlaps the evaluation window with the training data.
TEST_DAYS = 30

# Training baseline written by evaluate_models, read by monitoring
S3_TRAINING_BASELINE_KEY = "monitoring/training_baseline.json"

# Walk-forward backtest (multi-origin, robust) written by the training pipeline,
# read by the dashboard as the headline accuracy. Distinct from the single-holdout
# training baseline above.
S3_WALKFORWARD_KEY = "monitoring/walkforward_eval.json"

# MLflow S3 persistence (used by GitHub Actions to sync tracking db and artifacts)
S3_MLFLOW_DB_KEY = "mlflow/mlflow.db"

# Monitoring thresholds
PSI_MODERATE_THRESHOLD = 0.1
PSI_CRITICAL_THRESHOLD = 0.25
MAE_RETRAIN_MULTIPLIER = 1.5  # retrain if rolling MAE > 1.5x training MAE

# Retrain circuit breaker: never trigger a retrain more than once within this many
# days, even if MAE stays degraded. Without this, persistent degradation (or a
# training job that keeps failing before it clears the flag) would re-trigger
# training on every daily monitoring run. Trigger dates are kept in S3.
RETRAIN_COOLDOWN_DAYS = 7
S3_RETRAIN_HISTORY_KEY = "monitoring/retrain_history.json"

# SARIMAX order persistence. auto_arima re-searched every training cycle makes the
# production model architecture (and its coefficient panel) non-comparable month
# to month. The discovered order is cached in S3 and reused until it is older than
# SARIMAX_RESEARCH_DAYS, at which point the stepwise search runs again and re-pins.
S3_SARIMAX_ORDER_KEY = "monitoring/sarimax_order.json"
SARIMAX_RESEARCH_DAYS = 90
