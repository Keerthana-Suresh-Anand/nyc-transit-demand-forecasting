from datetime import date

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.dashboard.utils.data_loader import (
    load_forecast,
    load_history,
    load_past_forecasts_vs_actuals,
    load_pipeline_status,
    load_shap_image,
    load_weather_forecast,
)

st.set_page_config(
    page_title="NYC Transit Demand Forecast",
    page_icon="🚇",
    layout="wide",
    initial_sidebar_state="expanded",
)

_C = {
    "blue":   "#4c9be8",
    "orange": "#f4a261",
    "teal":   "#2a9d8f",
    "red":    "#e63946",
    "today":  "rgba(255,255,255,0.6)",
}

# ─── Data loading ─────────────────────────────────────────────────────────────
with st.spinner("Loading dashboard data..."):
    forecast_data = load_forecast()
    history = load_history(days=120)
    pipeline_status = load_pipeline_status()
    perf_df = load_past_forecasts_vs_actuals()
    weather_fc = load_weather_forecast()
    shap_img = load_shap_image()

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🚇 NYC Transit")
    st.caption("Weather-Driven Ridership Forecasting")
    st.divider()

    st.caption("Python · SARIMAX · XGBoost · MLflow · DVC · Docker · AWS S3 · GitHub Actions")
    st.divider()

    st.markdown("**Pipeline Health**")
    for status_key, date_key, label in [
        ("ingestion_status", "last_ingestion_date", "Ingestion"),
        ("training_status",  "last_training_date",  "Training"),
        ("prediction_status", "last_forecast_date", "Forecast"),
    ]:
        icon = {"success": "🟢", "failure": "🔴"}.get(
            pipeline_status.get(status_key) or "", "⚪"
        )
        run_date = pipeline_status.get(date_key) or "—"
        st.caption(f"{icon} {label} · {run_date}")

    st.divider()
    if forecast_data:
        sarimax_w = forecast_data.get("sarimax_weight", 0.6)
        xgb_w = forecast_data.get("xgboost_weight", 0.4)
        st.caption(f"Ensemble: SARIMAX {sarimax_w:.0%} + XGBoost {xgb_w:.0%}")

# ─── Page header ──────────────────────────────────────────────────────────────
st.title("NYC Subway Ridership Forecast")
st.markdown(
    "Daily NYC subway ridership forecasted 14 days ahead using weather data and an ensemble "
    "of SARIMAX and XGBoost, updated weekly via automated GitHub Actions pipelines."
)

if forecast_data is None or history is None:
    st.error(
        "Could not load forecast or historical data from S3. "
        "Verify AWS credentials and run the prediction pipeline."
    )
    st.stop()

fc_rows = pd.DataFrame(forecast_data["forecasts"])
fc_rows["date"] = pd.to_datetime(fc_rows["date"])

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — 14-DAY RIDERSHIP FORECAST
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("14-Day Ridership Forecast")

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=history.index,
    y=history["daily_ridership"] / 1_000_000,
    mode="lines",
    name="Actuals",
    line=dict(color=_C["blue"], width=2),
))

fig.add_trace(go.Scatter(
    x=pd.concat([fc_rows["date"], fc_rows["date"].iloc[::-1]]),
    y=pd.concat([fc_rows["ci_upper"], fc_rows["ci_lower"].iloc[::-1]]),
    fill="toself",
    fillcolor="rgba(180,180,180,0.15)",
    line=dict(color="rgba(0,0,0,0)"),
    hoverinfo="skip",
    name="95% CI",
))

fig.add_trace(go.Scatter(
    x=fc_rows["date"],
    y=fc_rows["sarimax_forecast_M"],
    mode="lines",
    name="SARIMAX",
    line=dict(color=_C["orange"], width=1.5, dash="dot"),
))

fig.add_trace(go.Scatter(
    x=fc_rows["date"],
    y=fc_rows["xgboost_forecast_M"],
    mode="lines",
    name="XGBoost",
    line=dict(color=_C["teal"], width=1.5, dash="dot"),
))

fig.add_trace(go.Scatter(
    x=fc_rows["date"],
    y=fc_rows["ensemble_forecast_M"],
    mode="lines+markers",
    name="Ensemble",
    line=dict(color=_C["red"], width=2.5),
    marker=dict(size=5),
))

last_actual_date = history.index.max().date()
today_date = date.today()

# Shade the MTA data lag zone between last actual and today
if last_actual_date < today_date:
    mid_ts = pd.Timestamp(last_actual_date) + (
        pd.Timestamp(today_date) - pd.Timestamp(last_actual_date)
    ) / 2
    fig.add_shape(
        type="rect",
        x0=str(last_actual_date), x1=str(today_date),
        y0=0, y1=1,
        xref="x", yref="paper",
        fillcolor="rgba(255,255,255,0.04)",
        line=dict(width=0),
    )
    fig.add_annotation(
        x=str(mid_ts), y=0.96,
        xref="x", yref="paper",
        text="MTA data lag",
        showarrow=False,
        font=dict(size=9, color="rgba(255,255,255,0.35)"),
        yanchor="top",
    )

fig.add_shape(
    type="line", x0=str(today_date), x1=str(today_date), y0=0, y1=1,
    xref="x", yref="paper",
    line=dict(color=_C["today"]),
)
fig.add_annotation(
    x=str(today_date), y=1, xref="x", yref="paper",
    text="Today", showarrow=False, yanchor="bottom", xanchor="right",
)

fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Ridership (Millions)",
    hovermode="x unified",
    height=460,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=50, r=30, t=40, b=50),
)
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — WEATHER AS A PREDICTIVE SIGNAL
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("Weather as a Predictive Signal")

hist_30 = history.iloc[-30:]
temp_col, precip_col, shap_col = st.columns([1, 1, 1])

with temp_col:
    fig_temp = go.Figure()
    if "temp" in hist_30.columns:
        fig_temp.add_trace(go.Scatter(
            x=hist_30.index, y=hist_30["temp"],
            mode="lines", name="Historical",
            line=dict(color=_C["blue"], dash="dot"),
        ))
    if weather_fc is not None and "temp" in weather_fc.columns:
        fig_temp.add_trace(go.Scatter(
            x=weather_fc["datetime"], y=weather_fc["temp"],
            mode="lines+markers", name="Forecast",
            line=dict(color=_C["orange"]),
        ))
    fig_temp.add_shape(
        type="line", x0=str(date.today()), x1=str(date.today()), y0=0, y1=1,
        xref="x", yref="paper", line=dict(color=_C["today"]),
    )
    fig_temp.update_layout(
        title="Temperature (°F)", height=340,
        margin=dict(t=40, b=30, l=40, r=20), hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_temp, use_container_width=True)

with precip_col:
    fig_precip = go.Figure()
    if "precip" in hist_30.columns:
        fig_precip.add_trace(go.Bar(
            x=hist_30.index, y=hist_30["precip"],
            name="Historical", marker_color=_C["blue"],
        ))
    if weather_fc is not None and "precip" in weather_fc.columns:
        fig_precip.add_trace(go.Bar(
            x=weather_fc["datetime"], y=weather_fc["precip"],
            name="Forecast", marker_color=_C["orange"],
        ))
    fig_precip.add_shape(
        type="line", x0=str(date.today()), x1=str(date.today()), y0=0, y1=1,
        xref="x", yref="paper", line=dict(color=_C["today"]),
    )
    fig_precip.update_layout(
        title="Precipitation (in)", height=340,
        barmode="overlay",
        margin=dict(t=40, b=30, l=40, r=20), hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_precip, use_container_width=True)

with shap_col:
    if shap_img:
        st.image(
            shap_img,
            caption="SHAP feature importance — which variables most influence the XGBoost forecast",
            use_container_width=True,
        )
    else:
        st.info("SHAP image not available. Run the training pipeline to generate and upload it.")

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — MODEL ACCURACY
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("Model Accuracy")

if perf_df is not None and len(perf_df) > 0:
    min_val = min(perf_df["actual_M"].min(), perf_df["ensemble_forecast_M"].min()) * 0.98
    max_val = max(perf_df["actual_M"].max(), perf_df["ensemble_forecast_M"].max()) * 1.02

    fig_scatter = go.Figure()
    fig_scatter.add_trace(go.Scatter(
        x=[min_val, max_val], y=[min_val, max_val],
        mode="lines",
        name="Perfect Forecast",
        line=dict(color="rgba(255,255,255,0.3)", dash="dash"),
        hoverinfo="skip",
    ))
    fig_scatter.add_trace(go.Scatter(
        x=perf_df["actual_M"],
        y=perf_df["ensemble_forecast_M"],
        mode="markers",
        name="Ensemble Forecast",
        marker=dict(color=_C["blue"], size=8, opacity=0.7),
        text=perf_df["date"].astype(str),
        hovertemplate="Date: %{text}<br>Actual: %{x:.3f}M<br>Forecast: %{y:.3f}M<extra></extra>",
    ))
    fig_scatter.update_layout(
        xaxis_title="Actual Ridership (M)",
        yaxis_title="Forecast Ridership (M)",
        height=420,
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=30, t=40, b=50),
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    m1, m2, m3 = st.columns(3)
    m1.metric("MAPE", f"{perf_df['abs_pct_error'].mean():.1f}%")
    m2.metric("MAE", f"{perf_df['error_M'].abs().mean():.3f} M")
    m3.metric("Forecast Runs", perf_df["forecast_run_date"].nunique())
else:
    st.info(
        "No forecast comparisons available yet. "
        "These accumulate weekly as forecasts age into the past."
    )
