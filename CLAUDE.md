# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

End-to-end ML system that forecasts daily NYC subway ridership 14 days ahead using weather data as exogenous features. Built as a portfolio project demonstrating production ML engineering practices.

**Models:** SARIMAX (captures weekly/annual seasonality with weather exog) + XGBoost (lag/calendar features, SHAP explainability). Predictions are blended into a weighted ensemble (60% SARIMAX / 40% XGBoost by default).

**Data sources:** MTA ridership via NYC Open Data SoDA API (updates Wednesdays with ~7-day lag, granularity: daily city-wide aggregate) · Visual Crossing weather API (historical + 14-day forecast) · Ticketmaster / NYC Open Data for major-venue event annotations.

**Infrastructure:** AWS S3 (three-layer data lake: bronze/silver/gold) · MLflow with local SQLite tracking, synced to S3 for CI · GitHub Actions (weekly ingestion, monthly retraining, daily monitoring, Docker publish on merge) · Streamlit dashboard deployed on Streamlit Community Cloud.

## Commands

```bash
# Install
pip install -e ".[pipeline,dev]"          # pipeline training + dev tools
pip install -e ".[dashboard]"             # dashboard only

# Lint
ruff check src/ tests/ pipelines/
ruff check src/ tests/ pipelines/ --fix

# Test
python -m pytest tests/unit/ -v           # all unit tests
python -m pytest tests/unit/test_drift_detector.py -v   # single file
python -m pytest tests/unit/ -k "psi"    # by keyword

# Run pipelines (from repo root)
python -m pipelines.run_ingestion
python -m pipelines.run_training
python -m pipelines.run_prediction
python -m pipelines.run_monitoring

# Dashboard
streamlit run src/dashboard/app.py

# MLflow UI
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Architecture

### Data flow

```
MTA API + Visual Crossing API + Ticketmaster API
        │
        ▼
    S3 bronze/          (raw CSVs per fetch window)
        │
        ▼
 data/silver/           (merged parquet, local)
        │
        ▼
 data/gold/             (two derived parquets, local)
   mta_sarima.parquet   ← daily city-wide aggregate + weather + holidays
   mta_ml.parquet       ← sarima parquet + lag/rolling features for XGBoost
        │
        ▼
  MLflow Registry       (SARIMAX champion + XGBoost champion, "Production" stage)
        │
        ▼
  S3 gold/forecasts/    (forecast_{date}.parquet + latest_forecast.json)
        │
        ▼
  Streamlit dashboard   (reads latest_forecast.json + gold parquets directly from S3)
```

### Module responsibilities

| Module | Reads from | Writes to |
|--------|-----------|-----------|
| `src/ingestion/ingest_mta.py` | MTA SoDA API | S3 bronze/mta/ |
| `src/ingestion/ingest_weather.py` | Visual Crossing API | S3 bronze/weather/ |
| `src/ingestion/ingest_events.py` | Ticketmaster / NYC Open Data | S3 bronze/events/ |
| `src/ingestion/merge_silver.py` | S3 bronze/ | data/silver/ local |
| `src/transformation/preprocess_sarima.py` | data/silver/ | data/gold/mta_sarima.parquet |
| `src/transformation/preprocess_ml.py` | data/gold/mta_sarima.parquet | data/gold/mta_ml.parquet |
| `src/training/train_sarimax.py` | data/gold/mta_sarima.parquet | MLflow registry |
| `src/training/train_xgboost.py` | data/gold/mta_ml.parquet | MLflow registry + reports/plots/ |
| `src/evaluation/evaluate_models.py` | MLflow registry | MLflow (promotes winner to Production) |
| `src/evaluation/drift_detector.py` | data/gold/mta_sarima.parquet | S3 monitoring/reports/ |
| `src/prediction/generate_forecast.py` | MLflow Production models + S3 bronze/weather/forecast/ | S3 gold/forecasts/ |
| `src/monitoring/monitor_performance.py` | S3 gold/forecasts/ + S3 gold/mta_sarima.parquet | S3 monitoring/reports/ + retrain_flag.json |
| `src/dashboard/utils/data_loader.py` | S3 (all layers) | — (cached reads) |

### Pipeline orchestrators

`pipelines/run_*.py` call the above modules in sequence and write a run log JSON to `S3 monitoring/pipeline_runs/{pipeline}_{date}.json`. Each orchestrator catches all exceptions, sets `status="failure"`, logs it to S3, then calls `sys.exit(1)`.

### Design decisions

**Why daily city-wide granularity (not station-level or hourly)**

The MTA network has ~500 stations. Station-level hourly forecasting means 500 × 24 = 12,000 simultaneous time series. With ~14 months of training data (from 2025-01-01), there is not enough history to fit reliable individual station models — the data requirement scales with granularity faster than the data grows. Additionally, the key exogenous signal in this project is weather, which affects total system demand (whether people take the subway at all) rather than which specific station they use at what hour. The weather signal weakens significantly at finer granularity. Daily city-wide is the right scope for the available data; station-level or line-level forecasting is the natural extension once 3+ years of history exist.

**Why MAE is the champion selection metric (not RMSE)**

RMSE penalises large errors more heavily, which sounds appealing — large forecast errors are costly. However, using RMSE for champion selection makes the result unstable: a model that gets one or two bad days in the 30-day holdout loses even if it is consistently better across the other 28 days. More importantly, the most operationally dangerous failure mode for a city-wide daily model is not occasional large spikes but systematic bias — consistently underpredicting by 2% every weekday is worse than occasional variance. MAE is more robust to outlier holdout days; mean signed error (also logged) catches systematic bias. RMSE is logged for reference but is not the decision metric.

### Key conventions

- **Config:** everything in `src/utils/config.py`. All S3 keys, local paths, API URLs, MLflow names, and thresholds are constants there. Never hardcode paths or env var names elsewhere.
- **Logging:** `logger = get_logger(__name__)` at the top of every module. Log pipeline start, data volumes, key decisions, and completion with summary stats.
- **S3 I/O:** always use helpers from `src/utils/s3_helpers.py` (`read_s3_parquet`, `write_s3_json`, `upload_s3_file`, etc.). Modules never call boto3 directly.
- **Entry points:** every src module exposes a `run()` function called by the pipeline orchestrator. Scripts can also be run directly via `python -m src.module.name`.
- **MLflow model names:** `sarimax_champion` and `xgboost_champion`. The prediction pipeline always loads the `Production` stage version.

### MLflow + GitHub Actions

MLflow uses a local SQLite backend (`mlflow.db`) and local artifact storage (`mlruns/`). GitHub Actions syncs both from S3 before running and uploads them back after training. One-time bootstrap after first local training run:

```bash
aws s3 cp mlflow.db s3://$BUCKET/mlflow/mlflow.db
aws s3 sync mlruns/ s3://$BUCKET/mlflow/mlruns/
```

### GitHub Actions secrets required

`AWS_ACCESS_KEY`, `AWS_SECRET_KEY`, `AWS_BUCKET_NAME`, `AWS_REGION`, `NY_APP_TOKEN`, `WEATHER_API_KEY`, `TICKETMASTER_API_KEY`, `DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN`

### Tests

`tests/conftest.py` sets dummy AWS env vars at module level (not via fixture) so that `src/utils/config.py` sees them when first imported during test collection. Tests never hit real AWS or APIs. The `xgboost_forecast` autoregressive loop is tested by mocking `mlflow.xgboost.load_model` and verifying that lag features for step N+1 come from step N's prediction, not from historical data.

### Dashboard data loading

`src/dashboard/utils/data_loader.py` wraps all S3 reads with `@st.cache_data(ttl=3600)`. The sidebar "Refresh data" button calls `st.cache_data.clear()`. The dashboard degrades gracefully when S3 data is unavailable — each loader returns `None` and the app shows an info/error banner instead of crashing.
