from datetime import date

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.dashboard.utils.data_loader import (
    load_drift_report,
    load_events,
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

# ─── Data loading ─────────────────────────────────────────────────────────────
forecast_data = load_forecast()
history = load_history(days=120)
drift = load_drift_report()
pipeline_status = load_pipeline_status()
events = load_events()

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🚇 NYC Transit")
    st.caption("Weather-Driven Ridership Forecasting")
    st.divider()

    if forecast_data:
        st.metric("Forecast Run Date", forecast_data.get("run_date", "—"))
        sarimax_w = forecast_data.get("sarimax_weight", 0.6)
        xgb_w = forecast_data.get("xgboost_weight", 0.4)
        st.caption(f"Ensemble: SARIMAX {sarimax_w:.0%} + XGBoost {xgb_w:.0%}")
    else:
        st.warning("No forecast data. Run the prediction pipeline first.")

    st.divider()

    if drift:
        psi_status = drift.get("psi_status", "unknown")
        icons = {"stable": "🟢", "moderate": "🟡", "critical": "🔴"}
        st.markdown(f"**Drift:** {icons.get(psi_status, '⚪')} {psi_status.title()}")
        st.caption(f"Max PSI: {drift.get('max_psi', 0):.3f} · {drift.get('report_date', '')}")

    st.divider()
    st.caption("Data: MTA Open Data + Visual Crossing")
    st.caption("Models: SARIMAX · XGBoost + SHAP")
    if st.button("🔄 Refresh data"):
        st.cache_data.clear()
        st.rerun()

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📈 Forecast", "📊 Model Performance", "⚙️ Pipeline Status"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — FORECAST
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    if forecast_data is None or history is None:
        st.error(
            "Could not load forecast or historical data from S3. "
            "Verify AWS credentials and run the prediction pipeline."
        )
        st.stop()

    fc_rows = pd.DataFrame(forecast_data["forecasts"])
    fc_rows["date"] = pd.to_datetime(fc_rows["date"])

    # KPI row — future_fc filters to rows that haven't passed yet
    tomorrow = pd.Timestamp(date.today()) + pd.Timedelta(days=1)
    future_fc = fc_rows[fc_rows["date"] >= tomorrow]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Latest Actual (M)", f"{history['daily_ridership'].iloc[-1] / 1_000_000:.2f}")
    if not future_fc.empty:
        k2.metric("Tomorrow Forecast (M)", f"{future_fc['ensemble_forecast_M'].iloc[0]:.2f}")
        k3.metric("7-Day Avg Forecast (M)", f"{future_fc['ensemble_forecast_M'].iloc[:7].mean():.2f}")
    else:
        k2.metric("Tomorrow Forecast (M)", "—")
        k3.metric("7-Day Avg Forecast (M)", "—")
    k4.metric("Data Through", str(history.index.max().date()))

    st.markdown("---")

    # Main ridership chart
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=history.index,
        y=history["daily_ridership"] / 1_000_000,
        mode="lines",
        name="Actuals",
        line=dict(color="#4c9be8", width=2),
    ))

    # CI band (from SARIMAX confidence interval)
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
        line=dict(color="#f4a261", width=1.5, dash="dot"),
    ))

    fig.add_trace(go.Scatter(
        x=fc_rows["date"],
        y=fc_rows["xgboost_forecast_M"],
        mode="lines",
        name="XGBoost",
        line=dict(color="#2a9d8f", width=1.5, dash="dot"),
    ))

    fig.add_trace(go.Scatter(
        x=fc_rows["date"],
        y=fc_rows["ensemble_forecast_M"],
        mode="lines+markers",
        name="Ensemble",
        line=dict(color="#e63946", width=2.5),
        marker=dict(size=5),
    ))

    # Event annotations — only within the visible chart window
    chart_start = history.index.min()
    chart_end = fc_rows["date"].max()
    for ev in events:
        ev_date = ev.get("date") or ev.get("datetime")
        ev_name = ev.get("name", "Event")
        if ev_date:
            x_val = str(ev_date)[:10]
            ev_ts = pd.Timestamp(x_val)
            if not (chart_start <= ev_ts <= chart_end):
                continue
            fig.add_shape(
                type="line", x0=x_val, x1=x_val, y0=0, y1=1,
                xref="x", yref="paper",
                line=dict(dash="dash", color="rgba(255,210,60,0.45)"),
            )
            fig.add_annotation(
                x=x_val, y=1, xref="x", yref="paper",
                text=ev_name[:18], showarrow=False,
                font=dict(size=9), yanchor="bottom",
            )

    fig.add_shape(
        type="line", x0=str(date.today()), x1=str(date.today()), y0=0, y1=1,
        xref="x", yref="paper",
        line=dict(color="rgba(255,255,255,0.6)"),
    )
    fig.add_annotation(
        x=str(date.today()), y=1, xref="x", yref="paper",
        text="Today", showarrow=False, yanchor="bottom", xanchor="right",
    )

    fig.update_layout(
        title="NYC Subway Daily Ridership — Historical & 14-Day Forecast",
        xaxis_title="Date",
        yaxis_title="Ridership (Millions)",
        hovermode="x unified",
        height=460,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=30, t=60, b=50),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Weather context
    st.subheader("Weather Context")
    weather_fc = load_weather_forecast()
    hist_30 = history.iloc[-30:]

    wc1, wc2 = st.columns(2)
    with wc1:
        fig_temp = go.Figure()
        if "temp" in hist_30.columns:
            fig_temp.add_trace(go.Scatter(
                x=hist_30.index, y=hist_30["temp"],
                mode="lines", name="Historical Temp",
                line=dict(color="#4c9be8", dash="dot"),
            ))
        if weather_fc is not None and "temp" in weather_fc.columns:
            fig_temp.add_trace(go.Scatter(
                x=weather_fc["datetime"], y=weather_fc["temp"],
                mode="lines+markers", name="Forecast Temp",
                line=dict(color="#f4a261"),
            ))
        fig_temp.add_shape(
            type="line", x0=str(date.today()), x1=str(date.today()), y0=0, y1=1,
            xref="x", yref="paper", line=dict(color="rgba(255,255,255,0.2)"),
        )
        fig_temp.update_layout(
            title="Temperature (°F)", height=240,
            margin=dict(t=40, b=30, l=40, r=20), hovermode="x unified",
        )
        st.plotly_chart(fig_temp, use_container_width=True)

    with wc2:
        fig_precip = go.Figure()
        if "precip" in hist_30.columns:
            fig_precip.add_trace(go.Bar(
                x=hist_30.index, y=hist_30["precip"],
                name="Historical Precip", marker_color="#4c9be8",
            ))
        if weather_fc is not None and "precip" in weather_fc.columns:
            fig_precip.add_trace(go.Bar(
                x=weather_fc["datetime"], y=weather_fc["precip"],
                name="Forecast Precip", marker_color="#f4a261",
            ))
        fig_precip.add_shape(
            type="line", x0=str(date.today()), x1=str(date.today()), y0=0, y1=1,
            xref="x", yref="paper", line=dict(color="rgba(255,255,255,0.2)"),
        )
        fig_precip.update_layout(
            title="Precipitation (in)", height=240,
            barmode="overlay",
            margin=dict(t=40, b=30, l=40, r=20), hovermode="x unified",
        )
        st.plotly_chart(fig_precip, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    pc1, pc2 = st.columns([1, 1])

    with pc1:
        st.subheader("Past Forecasts vs Actuals")
        perf_df = load_past_forecasts_vs_actuals()
        if perf_df is not None and len(perf_df) > 0:
            display = (
                perf_df[["date", "actual_M", "ensemble_forecast_M", "error_M", "abs_pct_error"]]
                .head(30)
                .rename(columns={
                    "date": "Date",
                    "actual_M": "Actual (M)",
                    "ensemble_forecast_M": "Forecast (M)",
                    "error_M": "Error (M)",
                    "abs_pct_error": "APE (%)",
                })
            )
            for col in ["Actual (M)", "Forecast (M)", "Error (M)"]:
                display[col] = display[col].round(3)
            display["APE (%)"] = display["APE (%)"].round(1)
            st.dataframe(display, use_container_width=True, hide_index=True)

            m1, m2 = st.columns(2)
            m1.metric("Mean MAPE", f"{perf_df['abs_pct_error'].mean():.1f}%")
            m2.metric("Mean |Error|", f"{perf_df['error_M'].abs().mean():.3f} M")
        else:
            st.info("No past forecast comparisons available yet. These accumulate weekly as forecasts age into the past.")

    with pc2:
        st.subheader("Weekly MAE")
        if perf_df is not None and len(perf_df) > 0:
            weekly = (
                perf_df.groupby("week")["error_M"]
                .apply(lambda x: x.abs().mean())
                .reset_index()
                .tail(12)
            )
            weekly["week"] = weekly["week"].astype(str)
            fig_mae = go.Figure(go.Bar(
                x=weekly["week"], y=weekly["error_M"], marker_color="#4c9be8",
            ))
            fig_mae.update_layout(
                yaxis_title="MAE (M)", xaxis_title="Week",
                height=300, margin=dict(t=20, b=50, l=40, r=20),
            )
            st.plotly_chart(fig_mae, use_container_width=True)
        else:
            st.info("Weekly MAE chart will populate as forecast history accumulates.")

    st.divider()

    # Drift monitor
    st.subheader("Data Drift Monitor")
    if drift:
        psi_scores = drift.get("psi_scores", {})
        psi_status = drift.get("psi_status", "unknown")
        icons = {"stable": "🟢", "moderate": "🟡", "critical": "🔴"}

        d1, d2, d3 = st.columns(3)
        d1.metric("Status", f"{icons.get(psi_status, '⚪')} {psi_status.title()}")
        d2.metric("Max PSI", f"{drift.get('max_psi', 0):.3f}")
        d3.metric("Report Date", drift.get("report_date", "—"))

        if psi_scores:
            psi_vals = list(psi_scores.values())
            colors = [
                "#e63946" if v > 0.25 else "#f4a261" if v > 0.1 else "#2a9d8f"
                for v in psi_vals
            ]
            fig_psi = go.Figure(go.Bar(
                x=list(psi_scores.keys()), y=psi_vals, marker_color=colors,
            ))
            fig_psi.add_hline(y=0.1, line_dash="dot", line_color="#f4a261",
                               annotation_text="Moderate (0.10)")
            fig_psi.add_hline(y=0.25, line_dash="dot", line_color="#e63946",
                               annotation_text="Critical (0.25)")
            fig_psi.update_layout(
                yaxis_title="PSI", height=260,
                margin=dict(t=20, b=30, l=40, r=20),
            )
            st.plotly_chart(fig_psi, use_container_width=True)
    else:
        st.info("No drift report available. Run the monitoring pipeline to generate one.")

    st.divider()

    # SHAP feature importance
    st.subheader("XGBoost Feature Importance (SHAP)")
    shap_img = load_shap_image()
    if shap_img:
        st.image(shap_img, caption="SHAP Summary Plot — XGBoost Champion", use_container_width=True)
    else:
        st.info("SHAP image not available. Run the training pipeline to generate and upload it.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — PIPELINE STATUS
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    def _badge(s: str | None) -> str:
        return {"success": "✅ Success", "failure": "❌ Failure"}.get(s or "", "— Not run")

    st.subheader("Pipeline Health")
    s1, s2, s3_col = st.columns(3)
    with s1:
        st.metric("Last Ingestion", pipeline_status.get("last_ingestion_date") or "—")
        st.caption(_badge(pipeline_status.get("ingestion_status")))
    with s2:
        st.metric("Last Training", pipeline_status.get("last_training_date") or "—")
        st.caption(_badge(pipeline_status.get("training_status")))
    with s3_col:
        st.metric("Last Forecast", pipeline_status.get("last_forecast_date") or "—")
        st.caption(_badge(pipeline_status.get("prediction_status")))

    st.divider()

    if history is not None:
        r1, r2 = st.columns(2)
        r1.metric("Gold Data Rows", f"{len(history):,}")
        r2.metric("Latest Actual Date", str(history.index.max().date()))

    st.divider()
    st.subheader("Data Freshness")
    st.caption(
        "MTA ridership data updates every Wednesday on data.ny.gov with a ~7-day lag. "
        "Weather data is fetched daily from Visual Crossing."
    )

    if forecast_data:
        run_ts = pd.Timestamp(forecast_data["run_date"])
        days_old = (pd.Timestamp(date.today()) - run_ts).days
        if days_old > 10:
            st.warning(f"⚠️ Forecast is {days_old} days old — consider triggering the prediction pipeline manually.")
        else:
            st.success(f"✅ Forecast is current ({days_old} day(s) old).")

    st.divider()
    st.subheader("Schedule")
    st.markdown("""
| Pipeline | Trigger | Frequency |
|----------|---------|-----------|
| Ingestion | Wednesday 14:00 UTC | Weekly |
| Training | First Wednesday of month | Monthly |
| Prediction | After ingestion / after training | Weekly |
| Monitoring | Daily 08:00 UTC | Daily |
""")
