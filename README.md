# NYC Transit Demand Forecasting

**Live dashboard:** https://nyc-transit-forecasting.streamlit.app/

An end-to-end ML system that produces a rolling 14-day forecast of daily NYC subway ridership, refreshed weekly. Weather was the starting hypothesis — does rain, snow, and temperature move ridership? It's included as a predictor and tested directly, and the evidence so far (SHAP, SARIMAX coefficient p-values, weak day-level correlations) says it's a **minor** one: recent ridership and the weekly calendar carry most of the signal. Built to demonstrate production ML engineering — automated pipelines, model registry, drift monitoring, a live dashboard — and honest, benchmark-driven evaluation.

---

## What it does

Every Wednesday, the MTA publishes updated ridership data with a ~7-day lag. This system:

1. **Ingests** new ridership + weather data automatically via GitHub Actions
2. **Trains** two models — SARIMAX and XGBoost — and selects the champion based on holdout MAE
3. **Forecasts** a rolling 14-day horizon using a weighted ensemble of both models
4. **Monitors** forecast accuracy and input data drift daily, triggering retraining when needed
5. **Displays** everything on a live Streamlit dashboard

---

## Why daily city-wide (not station-level or hourly)

The MTA has ~500 stations. Station-level hourly forecasting means fitting 12,000 simultaneous time series. With ~14 months of training data, there is not enough history to build reliable individual station models.

The modeling question here is about **system-level demand dynamics** — seasonality, the weekly calendar, holidays, and weather — which live at the daily city-wide level. Station-level ridership is dominated by fixed commuter patterns (Times Square is busy on a Tuesday morning regardless of the weather), which is a different problem. Any weather effect, if present, is also most detectable in the aggregate and washes out as granularity increases.

Daily city-wide is the right scope for the available data. Station-level or line-level forecasting is the natural extension with 3+ years of history.

---

## Models

### SARIMAX
Captures weekly and annual seasonality with weather exogenous variables (temperature, precipitation, snow lag, holidays). Best suited for structured seasonal patterns. Auto-ARIMA is used to select order parameters at each retraining.

### XGBoost
Uses lag features (ridership 1, 2, 3, 7, 14 days prior), rolling statistics (14-day average, 7-day std), and calendar features. SHAP values are computed at each run for explainability.

### Ensemble
Predictions are blended **50% SARIMAX + 50% XGBoost** (tunable in `src/utils/config.py`). Equal weights are a deliberate, evidence-based choice — see [Model evaluation](#model-evaluation) below.

### Champion selection
Both models are evaluated on the same 60-day holdout. Each family is promoted to `Production` in the MLflow registry only if the new version beats the current Production version of that family — so **both** SARIMAX and XGBoost live in `Production` simultaneously (the ensemble loads both); the better-performing family is recorded as champion metadata only. MAE is preferred over RMSE for promotion because RMSE is sensitive to individual bad holdout days, making selection unstable. Systematic bias (mean signed error) is also logged — consistent underprediction across weekdays is more operationally dangerous than occasional variance.

---

## Model evaluation

Reported accuracy is meaningless without two things: a **naive baseline** to beat, and an evaluation that **matches how the models are actually served**. Both are part of the evaluation.

### Baselines

| Benchmark | What it assumes |
|-----------|-----------------|
| Persistence (t-1) | Tomorrow equals today |
| Seasonal-naive (m=7) | Each day equals the same weekday last week |

Seasonal-naive is the hard-to-beat benchmark for daily ridership — weekly seasonality is the dominant pattern. Any model that doesn't clearly beat it isn't earning its complexity.

### Evaluation method

Models are scored with **14-day rolling-origin walk-forward** — the horizon and weekly re-anchoring cadence the system actually uses in production. A single long holdout would unfairly penalize XGBoost, whose recursive lag features compound error over long horizons it never serves; evaluating at the true 14-day horizon removes that artifact.

### Results (latest walk-forward, 11 weekly origins / 154 forecast points)

| Model | MAE (M) | MAPE |
|-------|---------|------|
| Persistence | 0.757 | 30.4% |
| Seasonal-naive (m=7) | 0.305 | 10.0% |
| SARIMAX | 0.258 | 9.3% |
| XGBoost | 0.255 | 8.7% |
| **Ensemble 50/50** | **0.248** | 8.8% |

Both models beat seasonal-naive by ~15%, justifying the modeling effort.

### Why 50/50 (and not a tuned weight)

Block-bootstrapped 95% confidence intervals on the pairwise MAE differences (10,000 resamples over whole origins) **all span zero** — SARIMAX vs XGBoost, and the ensemble vs either individual model, are **statistically indistinguishable** on the current data. When no model is reliably better, equal weighting is the honest choice; tuning a precise weight would overfit noise. A heavier weight will only be justified once a longer evaluation window (more origins, multiple seasons) tightens the intervals enough to show a real difference.

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

A single scrollable page — accessible at the Streamlit Community Cloud URL without any local setup:

> `requirements.txt` exists for Streamlit Community Cloud, which does not support pyproject.toml extras. All other environments use `pyproject.toml`.

- **Sidebar** — tech stack, pipeline health badges with last-run dates, and the active ensemble weights
- **Latest Ridership Forecast** — historical actuals + 14-day ensemble forecast with confidence intervals, individual SARIMAX and XGBoost lines, a shaded "MTA data lag" zone, and a today marker; captioned with the forecast's generation date and window
- **Weather as a Predictive Signal** — temperature-vs-ridership and precipitation-vs-ridership scatter plots with trend lines, demonstrating the weather signal directly
- **Model Accuracy** — predicted-vs-actual scatter against a perfect-forecast diagonal, XGBoost SHAP feature importance, and MAPE / MAE / forecast-run-count metrics

---

## Local setup

```bash
# Clone and install
git clone https://github.com/Keerthana-Suresh-Anand/nyc-transit-demand-forecasting
cd nyc-transit-demand-forecasting
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
- **Ensemble weights:** currently 50/50, chosen because the two models are statistically indistinguishable on the current evaluation window. A longer history (more walk-forward origins across multiple seasons) would tighten the confidence intervals and could justify an asymmetric weight.
- **Historical backfill:** current training data starts from January 2025. Incorporating 2023–2024 MTA ridership would extend the training window to 3 years and improve seasonal pattern estimation.
- **Docker-based pipeline execution:** GitHub Actions currently installs dependencies directly via `pip`. A more production-grade approach would have workflows pull and run the published Docker image, ensuring the CI environment is identical to any other deployment target.

---

## Tech stack

Python 3.12 · pandas · XGBoost · statsmodels · pmdarima · SHAP · MLflow · AWS S3 · Streamlit · Plotly · GitHub Actions · Docker
