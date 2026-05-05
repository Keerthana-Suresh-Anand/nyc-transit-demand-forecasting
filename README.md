# Weather-Driven Urban Transit Demand Forecasting

An end-to-end ML system that forecasts daily NYC subway ridership 14 days ahead using weather as an exogenous feature. Built to demonstrate production ML engineering practices: automated pipelines, model registry, drift monitoring, and a live dashboard.

---

## What it does

Every Wednesday, the MTA publishes updated ridership data with a ~7-day lag. This system:

1. **Ingests** new ridership + weather data automatically via GitHub Actions
2. **Trains** two models — SARIMAX and XGBoost — and selects the champion based on holdout MAE
3. **Forecasts** 14 days ahead using a weighted ensemble of both models
4. **Monitors** forecast accuracy and input data drift daily, triggering retraining when needed
5. **Displays** everything on a live Streamlit dashboard

---

## Why daily city-wide (not station-level or hourly)

The MTA has ~500 stations. Station-level hourly forecasting means fitting 12,000 simultaneous time series. With ~14 months of training data, there is not enough history to build reliable individual station models.

More importantly, the key signal in this project is **weather** — rain, snow, and temperature affect whether people choose to take the subway at all. That signal is strong at the system level and weakens significantly as granularity increases. Weather doesn't explain why Times Square station is busy on a Tuesday morning; commuter patterns do, and those are largely weather-independent.

Daily city-wide is the right scope for the available data. Station-level or line-level forecasting is the natural extension with 3+ years of history.

---

## Models

### SARIMAX
Captures weekly and annual seasonality with weather exogenous variables (temperature, precipitation, snow lag, holidays). Best suited for structured seasonal patterns. Auto-ARIMA is used to select order parameters at each retraining.

### XGBoost
Uses lag features (ridership 1, 2, 3, 7, 14 days prior), rolling statistics (14-day average, 7-day std), and calendar features. SHAP values are computed at each run for explainability.

### Ensemble
Predictions are blended: **60% SARIMAX + 40% XGBoost** (default weights, tunable in `src/utils/config.py`).

### Champion selection
Both models are evaluated on the same 30-day holdout set. The lower **MAE** model is promoted to `Production` in the MLflow registry; the other moves to `Staging`. MAE is preferred over RMSE for champion selection because RMSE is sensitive to individual bad holdout days, making selection unstable. Systematic bias (mean signed error) is also logged — consistent underprediction across weekdays is more operationally dangerous than occasional variance.

---

## Architecture

```
MTA API + Visual Crossing API + Ticketmaster API
        │
        ▼
    S3 bronze/              raw CSVs per fetch window
        │
        ▼
    S3 silver/              merged ridership + weather parquet
        │
        ▼
    S3 gold/                two derived parquets
      mta_sarima.parquet    daily city-wide aggregate + weather + holidays
      mta_ml.parquet        sarima data + lag/rolling features for XGBoost
        │
        ▼
    MLflow Registry         SARIMAX champion + XGBoost champion ("Production" stage)
        │
        ▼
    S3 gold/forecasts/      forecast_{date}.parquet + latest_forecast.json
        │
        ▼
    Streamlit dashboard     reads forecast JSON + gold parquets from S3
```

**Compute:** GitHub Actions (pipelines) + Streamlit Community Cloud (dashboard). AWS is storage only — no EC2, no SageMaker.

---

## Pipeline schedule

| Pipeline | Trigger | What it does |
|----------|---------|-------------|
| Ingestion | Every Wednesday 14:00 UTC | Fetches new MTA + weather + events data, updates silver on S3 |
| Training | First Wednesday of month | Retrains both models, selects champion, uploads gold + MLflow to S3 |
| Prediction | After ingestion or training | Generates 14-day ensemble forecast, writes to S3 |
| Monitoring | Daily 08:00 UTC | Checks forecast accuracy + PSI drift, triggers retraining if needed |
| Docker publish | On merge to main | Builds and pushes pipeline + dashboard images to Docker Hub |

---

## Monitoring and drift detection

**PSI (Population Stability Index)** is computed on the last 14 days of weather features vs a 90-day reference window:
- PSI < 0.10 → stable
- PSI 0.10–0.25 → moderate drift (logged, no action)
- PSI > 0.25 → critical drift → retrain flag written to S3 → training pipeline triggered

**Rolling MAE** is compared against training-time MAE. If rolling MAE exceeds 1.5× the training MAE, a retrain is also triggered.

---

## Dashboard

Three tabs — accessible at the Streamlit Community Cloud URL without any local setup:

- **Forecast** — 90-day historical actuals + 14-day ensemble forecast with confidence intervals, SARIMAX and XGBoost lines, NYC event annotations, weather context
- **Model Performance** — past forecasts vs actuals table, weekly MAE bar chart, PSI drift indicator, SHAP feature importance
- **Pipeline Status** — last run dates for each pipeline, data freshness indicator, schedule reference

---

## Local setup

```bash
# Clone and install
git clone https://github.com/KeerthanaSureshAnand/Weather-Driven-Urban-Transit-Demand-Forecasting
cd Weather-Driven-Urban-Transit-Demand-Forecasting
pip install -e ".[pipeline,dev]"

# Configure credentials
cp .env.example .env
# Fill in: AWS_ACCESS_KEY, AWS_SECRET_KEY, AWS_BUCKET_NAME, AWS_REGION,
#          NY_APP_TOKEN, WEATHER_API_KEY, TICKETMASTER_API_KEY
```

### First-time bootstrap (new S3 bucket)

```bash
python -m pipelines.run_ingestion     # fetch data → S3 bronze + silver
python -m pipelines.run_training      # train models → MLflow + S3 gold
aws s3 cp mlflow.db s3://YOUR_BUCKET/mlflow/mlflow.db
aws s3 sync mlruns/ s3://YOUR_BUCKET/mlflow/mlruns/
python -m pipelines.run_prediction    # generate first forecast → S3
```

After bootstrapping, GitHub Actions takes over on schedule.

### Viewing MLflow locally

After automated runs, sync from S3 to view experiment history:

```bash
aws s3 cp s3://YOUR_BUCKET/mlflow/mlflow.db ./mlflow.db
aws s3 sync s3://YOUR_BUCKET/mlflow/mlruns/ ./mlruns/
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

---

## GitHub Actions secrets required

`AWS_ACCESS_KEY` · `AWS_SECRET_KEY` · `AWS_BUCKET_NAME` · `AWS_REGION` · `NY_APP_TOKEN` · `WEATHER_API_KEY` · `TICKETMASTER_API_KEY` · `DOCKERHUB_USERNAME` · `DOCKERHUB_TOKEN`

---

## Known limitations and future work

- **Granularity:** daily city-wide forecasting is the right scope for 14 months of data. Line-level or station-level forecasting is the natural next step with 3+ years of history.
- **Event features:** NYC events (concerts, sports) are displayed as dashboard annotations but not yet used as model features — too sparse for a 460-day daily model.
- **MLflow hosting:** runs on local SQLite synced to S3. A production deployment would use a shared MLflow tracking server backed by PostgreSQL on RDS.
- **Ensemble weights:** 60/40 SARIMAX/XGBoost is a reasonable default. Weights should be tuned based on accumulated holdout performance after several months of automated runs.
- **Historical backfill:** current training data starts from January 2025. Incorporating 2023–2024 MTA ridership would extend the training window to 3 years and improve seasonal pattern estimation.
- **Docker-based pipeline execution:** GitHub Actions currently installs dependencies directly via `pip`. A more production-grade approach would have workflows pull and run the published Docker image, ensuring the CI environment is identical to any other deployment target.

---

## Tech stack

Python 3.12 · pandas · XGBoost · statsmodels · pmdarima · SHAP · MLflow · AWS S3 · Streamlit · Plotly · GitHub Actions · Docker
