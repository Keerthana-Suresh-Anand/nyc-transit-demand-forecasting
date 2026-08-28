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

The MTA has ~500 stations. Station-level (let alone hourly) forecasting means fitting hundreds to thousands of simultaneous, far sparser series — a fundamentally different and much more data- and compute-hungry problem than a single city-wide series.

The modeling question here is about **system-level demand dynamics** — seasonality, the weekly calendar, holidays, and weather — which live at the daily city-wide level. Station-level ridership is dominated by fixed commuter patterns (Times Square is busy on a Tuesday morning regardless of the weather), which is a different problem. Any weather effect, if present, is also most detectable in the aggregate and washes out as granularity increases.

Daily city-wide is the right scope for the modeling question here. Station- or line-level forecasting is a natural extension.

---

## Models

### SARIMAX
Captures weekly and annual seasonality with weather exogenous variables (temperature, precipitation, snow lag, holidays). Best suited for structured seasonal patterns. The `(p,d,q)` order is discovered by Auto-ARIMA and then **pinned** — cached to S3 and reused across retrains, and re-searched only when the cache is older than 90 days. This keeps the production architecture (and its coefficient panel) comparable month to month instead of silently changing shape every cycle, and the walk-forward backtest reads the same pinned order so it evaluates the architecture that actually ships.

### XGBoost
Uses lag features (ridership 1, 2, 3, 7, 14 days prior), rolling statistics (14-day average, 7-day std), and calendar features. SHAP values are computed at each run for explainability.

### Ensemble
Predictions are blended **50% SARIMAX + 50% XGBoost** (tunable in `src/utils/config.py`). Equal weights are a deliberate, evidence-based choice — see [Model evaluation](#model-evaluation) below.

### Champion selection
Both models are evaluated on a held-out 30-day tail (`TEST_DAYS` in `config.py`, shared by the trainers and the gate), then refit on the full dataset before registration so the shipped model uses every observation. Each family's `production` alias in the MLflow registry is moved to the new version only if its logged holdout MAE beats the currently aliased version of that family — so **both** SARIMAX and XGBoost carry a `production` alias simultaneously (the ensemble loads both); the better-performing family is recorded as champion metadata only. MAE is preferred over RMSE for promotion because RMSE is sensitive to individual bad holdout days, making selection unstable. Systematic bias (mean signed error) is also logged — consistent underprediction across weekdays is more operationally dangerous than occasional variance.

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

| Model | MAE (M) | MAPE | MASE |
|-------|---------|------|------|
| Persistence | 0.819 | 31.8% | 2.29 |
| Seasonal-naive (m=7) | 0.357 | 13.5% | 1.00 |
| SARIMAX | 0.260 | 10.3% | 0.73 |
| XGBoost | 0.219 | 8.7% | 0.61 |
| **Ensemble 50/50** | **0.216** | 8.9% | **0.60** |

The ensemble beats seasonal-naive by ~40%, justifying the modeling effort. Persistence is included
for completeness but is a weak benchmark on a series this seasonal — at MASE 2.29 it's over twice
the error of simply repeating last week, so seasonal-naive is the bar that matters.

### Why 50/50 (and not a tuned weight)

Block-bootstrapped 95% confidence intervals on the pairwise MAE differences (10,000 resamples over whole origins):

| Comparison | ΔMAE | 95% CI | Verdict |
|------------|------|--------|---------|
| SARIMAX vs XGBoost | +0.041 | [−0.005, +0.094] | tie |
| Ensemble vs XGBoost | −0.003 | [−0.025, +0.021] | tie |
| Ensemble vs SARIMAX | −0.044 | [−0.075, −0.016] | **ensemble better** |

XGBoost has the better point estimate, but the SARIMAX-vs-XGBoost interval still spans zero — no reliable winner between the two families, so equal weighting remains the honest choice; tuning a precise weight would overfit noise (the grid optimum, 0.35 SARIMAX, is within 0.003 MAE of the 50/50 point). The ensemble does **significantly** beat SARIMAX while tying XGBoost, so diversification demonstrably doesn't hurt and helps against one component — which is the case for keeping both. A heavier tilt toward XGBoost will only be justified once a longer window (more origins, multiple seasons) tightens the SARIMAX-vs-XGBoost interval.

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
    MLflow Registry         SARIMAX champion + XGBoost champion (@production alias)
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
| Ingestion | Every Wednesday 21:00 UTC | Fetches new MTA + weather + events data, updates silver on S3 |
| Training | First Wednesday of month | Retrains both models, selects champion, uploads gold + MLflow to S3 |
| Prediction | After ingestion or training | Generates 14-day ensemble forecast, writes to S3 |
| Monitoring | Daily 08:00 UTC | Checks forecast accuracy + PSI drift, triggers retraining if needed |
| Docker publish | On merge to main | Builds and pushes pipeline + dashboard images to Docker Hub |

---

## Monitoring and drift detection

**PSI (Population Stability Index)** is computed daily on the last 14 days of weather features vs a 90-day reference window and written to the drift report (visible on the dashboard):
- PSI < 0.10 → stable
- PSI 0.10–0.25 → moderate drift
- PSI > 0.25 → critical drift

PSI is **informational only — it never triggers retraining**. Weather features drift with the seasons, so PSI fires false alarms every spring and fall; retraining on those would churn models without improving accuracy.

**Rolling MAE is the sole retrain trigger:** if rolling forecast MAE exceeds 1.5× the training-time MAE, a retrain flag is written and the training pipeline is dispatched — subject to a 7-day cooldown circuit breaker so persistent degradation can't re-dispatch training on every daily run.

---

## Dashboard

A single scrollable page — accessible at the Streamlit Community Cloud URL without any local setup:

> `requirements.txt` exists for Streamlit Community Cloud, which does not support pyproject.toml extras. All other environments use `pyproject.toml`. The dashboard Docker image installs from the same `requirements.txt`, so the published container is a faithful replica of the live deployment — it isn't used to serve the dashboard today (Streamlit Cloud hosts it for free), but it keeps the app portable to any container platform (Cloud Run, ECS) if that changes.

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

- **Forward reach:** the MTA publishes ridership with a ~1–2 week lag, so only the latter part of each 14-day forecast is genuinely ahead of today — the earlier days fill the gap to the last published week.
- **One-off shocks:** the models learn regular patterns (weekly seasonality, holidays, weather) and can't anticipate service disruptions, special events, or structural breaks until they appear in the data.
- **Evaluation power:** the walk-forward backtest uses a fixed 90-day / 11-origin window (production-faithful — it mirrors how the model serves between retrains), so its confidence intervals are wide and it spans only one season. A longer, multi-season window with a per-origin refit would tighten the CIs and could justify an asymmetric ensemble weight (currently 50/50 because the two models are statistically indistinguishable on this window).
- **Ensemble uncertainty:** the forecast is a point estimate — a calibrated prediction band around the ensemble is planned (the shipped SARIMAX interval isn't the ensemble's).
- **Granularity:** daily city-wide is the scope; line- or station-level forecasting is a natural extension (a different, far more data- and compute-hungry problem).
- **Event features:** NYC events (concerts, sports) are shown as dashboard context but not yet used as model features — too sparse to help a daily city-wide model.
- **MLflow hosting:** runs on local SQLite synced to S3; a production deployment would use a shared tracking server on PostgreSQL/RDS.
- **Docker in CI:** the CI workflow installs dependencies via `pip`; running the published Docker image in CI would make the environment identical across targets (the training/prediction pipelines already run the image).

---

## Tech stack

Python 3.12 · pandas · XGBoost · statsmodels · pmdarima · SHAP · MLflow · AWS S3 · Streamlit · Plotly · GitHub Actions · Docker
