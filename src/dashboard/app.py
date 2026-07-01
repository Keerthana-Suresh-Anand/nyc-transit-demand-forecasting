from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.dashboard.utils.data_loader import (
    load_forecast,
    load_history,
    load_past_forecasts_vs_actuals,
    load_pipeline_status,
    load_sarimax_coefficients,
    load_shap_image,
    load_training_baseline,
    load_walkforward,
)

st.set_page_config(
    page_title="NYC Transit Demand Forecast",
    page_icon="🚇",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Brighten the muted grey sidebar caption text for readability (bold headers
# stay full-white, so the visual hierarchy is preserved).
st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] small,
    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p,
    section[data-testid="stSidebar"] .stCaption,
    section[data-testid="stSidebar"] .stCaption p {
        color: rgba(255, 255, 255, 0.95) !important;
    }
    /* Pin the SHAP image to the same height as the paired SARIMAX coefficient
       chart (460px) so the two interpretability panels align. object-fit:contain
       scales without distortion; the PNG's dark background hides any letterbox.
       The SHAP column is widened below so the image fills more of this box. */
    [data-testid="stImage"] img {
        height: 460px !important;
        object-fit: contain;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

_C = {
    "blue":   "#4c9be8",
    "orange": "#f4a261",
    "teal":   "#2a9d8f",
    "red":    "#e63946",
    "today":  "rgba(255,255,255,0.6)",
}

# Hide Plotly's hover toolbar entirely — these are presentational charts, not an
# analysis sandbox, and nobody downloads PNGs from a portfolio. Tooltips are
# independent of the toolbar, so hovering still shows values.
_PLOTLY_CONFIG = {"displayModeBar": False}


def _show(fig):
    """Render a chart with pan/zoom locked. These are fixed, curated views, so
    fixedrange disables drag-pan and scroll-zoom (which the hidden modebar can't
    undo) while leaving hover tooltips intact."""
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    st.plotly_chart(fig, use_container_width=True, config=_PLOTLY_CONFIG)

# Medallion data-flow diagram for the "How It Works" section (rendered by graphviz).
_PIPELINE_DOT = """
digraph {
    rankdir=LR; bgcolor="transparent";
    node [shape=box style="rounded,filled" fontcolor=white color="#444444" fillcolor="#1b2330" fontsize=10];
    edge [color="#888888"];
    apis   [label="MTA · Weather · Events\\nAPIs" fillcolor="#22303f"];
    bronze [label="Bronze\\nraw CSVs"];
    silver [label="Silver\\nmerged"];
    gold   [label="Gold\\nfeatures"];
    mlflow [label="MLflow\\nRegistry" fillcolor="#2a3b2f"];
    fc     [label="Forecasts\\nS3"];
    dash   [label="Dashboard" fillcolor="#3a2b3f"];
    apis -> bronze -> silver -> gold -> mlflow -> fc -> dash;
}
"""

# ─── Data loading ─────────────────────────────────────────────────────────────
with st.spinner("Loading dashboard data..."):
    forecast_data = load_forecast()
    history = load_history(days=365)
    pipeline_status = load_pipeline_status()
    perf_df = load_past_forecasts_vs_actuals()
    shap_img = load_shap_image()
    sarimax_coef = load_sarimax_coefficients()
    baseline = load_training_baseline()
    walkforward = load_walkforward()


def _accuracy_view(wf: dict | None, bl: dict | None) -> dict | None:
    """Unified accuracy source: prefer the multi-origin walk-forward backtest;
    fall back to the single 30-day holdout when the walk-forward hasn't run yet."""
    if wf and wf.get("mae"):
        m = wf["mae"]
        return {
            "source": "walk-forward",
            "label": "Backtest MAE",
            "ens_mae": m.get("ensemble_50_50"),
            "sarimax_mae": m.get("sarimax"),
            "xgboost_mae": m.get("xgboost"),
            "seasonal_naive_mae": m.get("seasonal_naive"),
            "persistence_mae": m.get("persistence"),
            "n_origins": wf.get("n_origins"),
            "significance": wf.get("significance"),
            "help": f"Ensemble MAE over a {wf.get('n_origins', '?')}-origin, 14-day rolling "
                    "walk-forward backtest (matches production's horizon and weekly re-anchor).",
        }
    if bl and bl.get("ensemble_mae") is not None:
        b = bl.get("baselines", {})
        return {
            "source": "holdout",
            "label": "Holdout MAE",
            "ens_mae": bl.get("ensemble_mae"),
            "sarimax_mae": bl.get("sarimax_mae"),
            "xgboost_mae": bl.get("xgboost_mae"),
            "seasonal_naive_mae": (b.get("seasonal_naive_m7") or {}).get("mae"),
            "persistence_mae": (b.get("persistence") or {}).get("mae"),
            "n_origins": None,
            "significance": None,
            "help": "Ensemble MAE on the reserved 30-day holdout (single split).",
        }
    return None


acc = _accuracy_view(walkforward, baseline)


def side_text(text: str) -> None:
    """Bright sidebar caption. st.caption renders too dim on the dark theme, and
    global <style> targeting Streamlit's caption testid isn't applying in this
    version — so style the text inline, which always wins."""
    st.markdown(
        f"<div style='color:#bfc3ca;font-size:0.85rem;line-height:1.55;'>{text}</div>",
        unsafe_allow_html=True,
    )


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🚇 NYC Transit")
    side_text("Weather-Driven Ridership Forecasting")
    st.divider()

    side_text("Python · SARIMAX · XGBoost · MLflow · DVC · Docker · AWS S3 · GitHub Actions")
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
        side_text(f"{icon} {label} · {run_date}")

    st.divider()

    data_through = pipeline_status.get("data_through_date")
    if data_through:
        # Data coverage, not a run date: MTA publishes with a lag, so this trails
        # the ingestion run date above by ~a week.
        st.markdown("**Data Coverage**")
        caption = f"📅 Through · {data_through}"
        try:
            lag_days = (date.today() - date.fromisoformat(data_through)).days
            caption += f" · {lag_days}d behind"
        except ValueError:
            pass
        side_text(caption)
        st.divider()

    if forecast_data:
        sarimax_w = forecast_data.get("sarimax_weight", 0.5)
        xgb_w = forecast_data.get("xgboost_weight", 0.5)
        side_text(f"Ensemble: SARIMAX {sarimax_w:.0%} + XGBoost {xgb_w:.0%}")

# ─── Page header ──────────────────────────────────────────────────────────────
st.title("NYC Subway Ridership Forecast", anchor=False)
st.markdown(
    "**Decision support for service planning** — a rolling **14-day ridership forecast** for NYC "
    "subway demand, refreshed weekly. A **SARIMAX + XGBoost** ensemble over ridership history, "
    "day-of-week, holidays, and weather, automatically retrained and monitored by scheduled "
    "**GitHub Actions** pipelines."
)

if forecast_data is None or history is None:
    st.error(
        "Could not load forecast or historical data from S3. "
        "Verify AWS credentials and run the prediction pipeline."
    )
    st.stop()

fc_rows = pd.DataFrame(forecast_data["forecasts"])
fc_rows["date"] = pd.to_datetime(fc_rows["date"])

# ── Provenance: which registered models + training date produced this forecast ──
_mv = forecast_data.get("model_versions") or {}


def _ver(name: str) -> str:
    d = _mv.get(name)
    return f"v{d['version']}" if isinstance(d, dict) and d.get("version") is not None else "—"


_sar_w = int(round(forecast_data.get("sarimax_weight", 0.5) * 100))
_trained = pipeline_status.get("last_training_date") or "—"
st.caption(
    f"🤖 SARIMAX {_ver('sarimax')} · XGBoost {_ver('xgboost')} · "
    f"ensemble {_sar_w}/{100 - _sar_w} · models trained {_trained}"
)

# ── KPI hero row ────────────────────────────────────────────────────────────────
k1, k2, k3 = st.columns(3)

# Forward 7 days of the 14-day window (the most-forward week), reported as a total
# with its explicit date range — states a horizon, not a misleading "future" claim,
# given MTA's ~1-week lag. fwd7 / fwd7_range are reused by the daily table below.
fc_sorted = fc_rows.sort_values("date")
fwd7 = fc_sorted.tail(7)
_d0, _d1 = fwd7["date"].iloc[0], fwd7["date"].iloc[-1]
if _d0.month == _d1.month:
    fwd7_range = f"{_d0.strftime('%b')} {_d0.day}–{_d1.day}"
else:
    fwd7_range = f"{_d0.strftime('%b')} {_d0.day} – {_d1.strftime('%b')} {_d1.day}"
total7 = float(fwd7["ensemble_forecast_M"].sum())
wow_txt = None
try:
    prior7 = history["daily_ridership"].tail(7) / 1_000_000
    if len(prior7) == 7:
        base = float(prior7.sum())
        wow_txt = f"{(total7 - base) / base * 100:+.1f}% vs prior 7d"
except Exception:
    pass
k1.metric(
    f"7-Day Forecast · {fwd7_range}", f"{total7:.1f}M", wow_txt,
    help="Total predicted ridership over the forecast's forward 7 days. MTA's ~1-week lag means "
         "these are the most-forward days of the 14-day window; the date range shows exactly which.",
)

if acc and acc["ens_mae"] is not None and acc["seasonal_naive_mae"] is not None:
    ens_mae, sn_mae = acc["ens_mae"], acc["seasonal_naive_mae"]
    k2.metric(acc["label"], f"{ens_mae:.3f}M", help=acc["help"])
    k3.metric("Beats Seasonal-Naive", f"{(sn_mae - ens_mae) / sn_mae * 100:.0f}%",
              help="Ensemble MAE vs the same-weekday-last-week benchmark on the same evaluation.")
else:
    k2.metric("Backtest MAE", "—")
    k3.metric("Beats Seasonal-Naive", "—")

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — RIDERSHIP FORECAST
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("Latest Ridership Forecast", anchor=False)
run_date = forecast_data.get("run_date", "—")
# Cross-platform date format: %-d (no-leading-zero day) is glibc-only and breaks
# on Windows, so build it from .day instead.
_fc_min, _fc_max = fc_rows["date"].min(), fc_rows["date"].max()
fc_start = f"{_fc_min.strftime('%b')} {_fc_min.day}"
fc_end = f"{_fc_max.strftime('%b')} {_fc_max.day}"
st.caption(f"Generated {run_date} · {fc_start}–{fc_end}")

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=history.index,
    y=history["daily_ridership"] / 1_000_000,
    mode="lines",
    name="Actuals",
    line=dict(color=_C["blue"], width=2),
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
    x=str(today_date), y=0.02, xref="x", yref="paper",
    text="Today", showarrow=False, yanchor="bottom", xanchor="left",
)

fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Ridership (Millions)",
    hovermode="x unified",
    height=460,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=50, r=30, t=40, b=50),
)
_show(fig)

# Per-day numbers behind the forward-7 KPI — the concrete deliverable.
st.markdown(f"**Daily forecast · {fwd7_range}**")
_daily = pd.DataFrame({
    "Date": fwd7["date"].dt.strftime("%a, %b %d"),
    "Forecast (M)": fwd7["ensemble_forecast_M"].round(2),
})
st.dataframe(_daily, hide_index=True, use_container_width=True)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — MODEL ACCURACY
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("Model Accuracy", anchor=False)
if acc and acc["source"] == "walk-forward":
    st.caption(f"Robust **{acc['n_origins']}-origin walk-forward backtest** (14-day horizon, weekly "
               "re-anchor — how the model is actually used), plus live tracking as forecasts age into actuals.")
else:
    st.caption("Evaluated on a reserved 30-day holdout, then tracked live as past forecasts age into actuals.")

# ── Models vs naive baselines — the 'compared to what?' ───────────────────────────
if acc:
    _bars = [
        ("Persistence (t−1)", acc["persistence_mae"], "base"),
        ("Seasonal-naive (m=7)", acc["seasonal_naive_mae"], "base"),
        ("SARIMAX", acc["sarimax_mae"], "model"),
        ("XGBoost", acc["xgboost_mae"], "model"),
        ("Ensemble", acc["ens_mae"], "ensemble"),
    ]
    _bars = [b for b in _bars if b[1] is not None]
    if _bars:
        _cmap = {"base": "rgba(255,255,255,0.35)", "model": _C["teal"], "ensemble": _C["red"]}
        fig_bl = go.Figure(go.Bar(
            x=[b[1] for b in _bars], y=[b[0] for b in _bars], orientation="h",
            marker=dict(color=[_cmap[b[2]] for b in _bars]),
            text=[f"{b[1]:.3f}" for b in _bars], textposition="auto",
            hovertemplate="%{y}: MAE %{x:.3f} M<extra></extra>",
        ))
        _bar_title = "Backtest MAE" if acc["source"] == "walk-forward" else "Holdout MAE"
        fig_bl.update_layout(
            title=f"{_bar_title} — lower is better (models vs naive benchmarks)",
            xaxis_title="MAE (millions of riders)",
            height=300, margin=dict(l=10, r=20, t=40, b=40),
            yaxis=dict(autorange="reversed"),
        )
        _show(fig_bl)
        st.caption("Seasonal-naive (same weekday last week) is the benchmark to beat — weekly "
                   "seasonality dominates daily ridership, so a model must clear it to earn its complexity.")

    # Bootstrap significance — only the walk-forward has enough origins to resample.
    if acc.get("significance"):
        st.markdown("**Is the difference real?** — block bootstrap, 95% CI on the MAE difference:")
        st.caption("When two models are statistically indistinguishable (CI spans zero), neither "
                   "earns a heavier weight — which is why the ensemble weights them 50/50.")
        for _key, _lbl in [
            ("sarimax_vs_xgboost", "SARIMAX vs XGBoost"),
            ("ensemble_vs_sarimax", "Ensemble vs SARIMAX"),
            ("ensemble_vs_xgboost", "Ensemble vs XGBoost"),
        ]:
            s = acc["significance"].get(_key)
            if s:
                st.caption(f"• {_lbl}: **{s['verdict']}** — ΔMAE {s['dmae']:+.3f}, "
                           f"95% CI [{s['ci_lo']:+.3f}, {s['ci_hi']:+.3f}]")

# ── Per-model metrics (walk-forward only — needs bias/MASE the backtest computes) ──
if walkforward and walkforward.get("bias"):
    _mae_d = walkforward.get("mae", {})
    _mase_d = walkforward.get("mase", {})
    _bias_d = walkforward["bias"]
    _rows = []
    for _key, _name in [
        ("seasonal_naive", "Seasonal-naive"), ("persistence", "Persistence"),
        ("sarimax", "SARIMAX"), ("xgboost", "XGBoost"), ("ensemble_50_50", "Ensemble"),
    ]:
        if _key in _mae_d:
            _rows.append({
                "Model": _name,
                "MAE (M)": round(_mae_d[_key], 3),
                "MASE": round(_mase_d[_key], 2) if _key in _mase_d else None,
                "Bias (M)": round(_bias_d[_key], 3) if _key in _bias_d else None,
            })
    st.divider()
    st.markdown("**Per-model metrics** (walk-forward)")
    st.dataframe(pd.DataFrame(_rows), hide_index=True, use_container_width=True)
    st.caption("MASE < 1 beats seasonal-naive. Bias = mean(forecast − actual): "
               "**+ over-forecasts, − under-forecasts** (systematic skew, which MAE/MAPE hide).")

# ── Live accuracy ──────────────────────────────────────────────────────────────────
st.divider()
st.markdown("**Live tracking** — realized error of served forecasts as they age into actuals")
if perf_df is not None and len(perf_df) > 0:
    c1, c2 = st.columns(2)
    c1.metric("Live MAPE", f"{perf_df['abs_pct_error'].mean():.1f}%")
    c2.metric("Live MAE", f"{perf_df['error_M'].abs().mean():.3f}M")
    st.caption(
        f"Live = realized error of recently-served forecasts, over "
        f"{len(perf_df)} forecast-days across {perf_df['forecast_run_date'].nunique()} runs. "
        "Includes model versions since improved — expected to run above the backtest and "
        "converge toward it as current-model forecasts age into actuals."
    )

    col_h, col_s = st.columns(2)

    with col_h:
        by_h = (
            perf_df.assign(abs_err=perf_df["error_M"].abs())
            .groupby("horizon")["abs_err"].mean().reset_index().sort_values("horizon")
        )
        fig_h = go.Figure(go.Scatter(
            x=by_h["horizon"], y=by_h["abs_err"],
            mode="lines+markers", line=dict(color=_C["red"], width=2),
            hovertemplate="Day %{x}: MAE %{y:.3f} M<extra></extra>",
        ))
        fig_h.update_layout(
            title="Error grows with horizon",
            xaxis_title="Days ahead", yaxis_title="Mean abs error (M)",
            height=360, margin=dict(l=50, r=20, t=40, b=45),
        )
        _show(fig_h)
        st.caption("Forecast error by lead time — day 1 is easy, day 14 compounds. This is the "
                   "horizon the model actually serves in production.")

    with col_s:
        min_val = min(perf_df["actual_M"].min(), perf_df["ensemble_forecast_M"].min()) * 0.98
        max_val = max(perf_df["actual_M"].max(), perf_df["ensemble_forecast_M"].max()) * 1.02
        fig_scatter = go.Figure()
        fig_scatter.add_trace(go.Scatter(
            x=[min_val, max_val], y=[min_val, max_val],
            mode="lines", name="Perfect",
            line=dict(color="rgba(255,255,255,0.3)", dash="dash"), hoverinfo="skip",
        ))
        fig_scatter.add_trace(go.Scatter(
            x=perf_df["actual_M"], y=perf_df["ensemble_forecast_M"],
            mode="markers", name="Forecast",
            marker=dict(color=_C["blue"], size=8, opacity=0.7),
            text=perf_df["date"].astype(str),
            hovertemplate="%{text}<br>Actual %{x:.3f}M<br>Forecast %{y:.3f}M<extra></extra>",
        ))
        fig_scatter.update_layout(
            title="Forecast vs actual",
            xaxis_title="Actual (M)", yaxis_title="Forecast (M)",
            height=360, hovermode="closest",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=50, r=20, t=40, b=45),
        )
        _show(fig_scatter)
        st.caption("Points on the dashed line are perfect; tight clustering means low bias.")
else:
    st.info("Live accuracy accumulates weekly as forecasts age into actuals — check back after the next cycle.")

# ── Interpretability (training-derived; shown regardless of live history) ─────────
st.divider()
st.markdown("**How each model reasons** — interpretability for both halves of the ensemble")
# SHAP gets a wider column: it's a dense beeswarm that needs more horizontal
# room to stay legible than the SARIMAX bar chart.
shap_col, coef_col = st.columns([1.3, 1])

with shap_col:
    if shap_img:
        st.image(
            shap_img,
            caption="XGBoost — SHAP feature importance: ridership-lag momentum dominates; weather adds signal at the margin",
            use_container_width=True,
        )
    else:
        st.info("XGBoost SHAP plot generates on the next training run.")

with coef_col:
    if sarimax_coef and sarimax_coef.get("exog_coefficients"):
        _LABELS = {"temp": "Temperature", "precip": "Precipitation",
                   "snow_lag1": "Snow (prev day)", "is_holiday": "Holiday"}
        recs = sarimax_coef["exog_coefficients"]
        names = [_LABELS.get(r["feature"], r["feature"]) for r in recs]
        vals = [r["coefficient"] for r in recs]
        colors = [_C["teal"] if v >= 0 else _C["red"] for v in vals]
        opac = [1.0 if r["p_value"] < 0.05 else 0.4 for r in recs]
        fig_coef = go.Figure(go.Bar(
            x=vals, y=names, orientation="h",
            marker=dict(color=colors, opacity=opac),
            hovertemplate="%{y}: %{x:.3f}<extra></extra>",
        ))
        fig_coef.add_shape(
            type="line", x0=0, x1=0, y0=-0.5, y1=len(names) - 0.5,
            line=dict(color="rgba(255,255,255,0.3)"),
        )
        fig_coef.update_layout(
            height=460,
            xaxis_title="Coefficient on scaled input (+ raises / − lowers ridership)",
            margin=dict(l=10, r=20, t=30, b=50),
        )
        _show(fig_coef)
        st.caption("SARIMAX — weather/holiday effects (inputs scaled 0–1, so bars are comparable). "
                   "Faded bars are not significant (p ≥ 0.05).")
    else:
        st.info("SARIMAX coefficients generate on the next training run.")

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — WEATHER AS A PREDICTIVE SIGNAL
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("Does Weather Predict Ridership?", anchor=False)
st.caption("Weather correlates only weakly with daily ridership — recent ridership and the "
           "weekly calendar carry most of the predictive signal.")

hist_wx = history.dropna(subset=["temp", "precip", "daily_ridership"]).copy()
ridership_M = hist_wx["daily_ridership"] / 1_000_000
temp_col, precip_col = st.columns([1, 1])

with temp_col:
    z = np.polyfit(hist_wx["temp"], ridership_M, 1)
    x_line = np.linspace(hist_wx["temp"].min(), hist_wx["temp"].max(), 100)
    fig_temp = go.Figure()
    fig_temp.add_trace(go.Scatter(
        x=hist_wx["temp"], y=ridership_M,
        mode="markers", name="Daily",
        marker=dict(color=_C["blue"], size=5, opacity=0.45),
        hovertemplate="Temp: %{x:.1f}°C<br>Ridership: %{y:.2f}M<extra></extra>",
    ))
    fig_temp.add_trace(go.Scatter(
        x=x_line, y=np.poly1d(z)(x_line),
        mode="lines", name="Trend",
        line=dict(color=_C["orange"], width=2),
        hoverinfo="skip",
    ))
    fig_temp.update_layout(
        title="Temperature vs Ridership",
        xaxis_title="Temperature (°C)", yaxis_title="Ridership (M)",
        height=340, margin=dict(t=40, b=30, l=40, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    _r_t = float(np.corrcoef(hist_wx["temp"], ridership_M)[0, 1])
    _show(fig_temp)
    st.caption(f"Pearson r = {_r_t:+.2f} — weak linear correlation")

with precip_col:
    z = np.polyfit(hist_wx["precip"], ridership_M, 1)
    x_line = np.linspace(hist_wx["precip"].min(), hist_wx["precip"].max(), 100)
    fig_precip = go.Figure()
    fig_precip.add_trace(go.Scatter(
        x=hist_wx["precip"], y=ridership_M,
        mode="markers", name="Daily",
        marker=dict(color=_C["blue"], size=5, opacity=0.45),
        hovertemplate="Precip: %{x:.2f} mm<br>Ridership: %{y:.2f}M<extra></extra>",
    ))
    fig_precip.add_trace(go.Scatter(
        x=x_line, y=np.poly1d(z)(x_line),
        mode="lines", name="Trend",
        line=dict(color=_C["orange"], width=2),
        hoverinfo="skip",
    ))
    fig_precip.update_layout(
        title="Precipitation vs Ridership",
        xaxis_title="Precipitation (mm)", yaxis_title="Ridership (M)",
        height=340, margin=dict(t=40, b=30, l=40, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    _r_p = float(np.corrcoef(hist_wx["precip"], ridership_M)[0, 1])
    _show(fig_precip)
    st.caption(f"Pearson r = {_r_p:+.2f} — weak linear correlation")

st.caption(
    "⚠️ Raw daily correlations — confounded by season and day-of-week (e.g. cold months "
    "are also winter-schedule months). Shown to illustrate the *marginal* weather signal, not "
    "a causal effect; the models isolate it by controlling for calendar features."
)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — HOW IT WORKS
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("How It Works", anchor=False)
st.caption("The system behind the numbers — built and evaluated like a production service.")
arch_col, method_col = st.columns(2)

with arch_col:
    with st.expander("🏗  Architecture & MLOps"):
        st.graphviz_chart(_PIPELINE_DOT)
        st.markdown(
            "- **Bronze → Silver → Gold** medallion layers in S3 (raw → merged → modelled features)\n"
            "- **MLflow** registry — both models versioned, gated to Production on holdout MAE\n"
            "- **GitHub Actions** runs all compute: 4 scheduled pipelines (ingest / train / predict / monitor)\n"
            "- **Docker** image for reproducible runs · **DVC** snapshots every gold dataset\n"
            "- **Self-monitoring** — rolling MAE triggers retrains, with a cooldown to prevent thrash\n"
            "- AWS is storage-only; no servers to maintain"
        )

with method_col:
    with st.expander("🔬  Evaluation methodology"):
        st.markdown(
            "**Why 50/50?** A 14-day rolling-origin walk-forward (matching production's horizon and "
            "weekly re-anchor) found SARIMAX and XGBoost statistically indistinguishable — a block "
            "bootstrap put every pairwise MAE-difference 95% CI across zero. With no reliable winner, "
            "equal weighting is the honest choice; tuning a precise weight overfits noise.\n\n"
            "**Why keep both?** The ensemble's point estimate is best and the weight curve is convex "
            "(genuine diversification), with no evidence it hurts.\n\n"
            "**Retraining** is triggered by MAE degradation, not drift — PSI on weather features fires "
            "seasonal false alarms, so it's tracked but informational only.\n\n"
            "**Benchmarks** (seasonal-naive, persistence) are scored every cycle: a model is only as "
            "good as what it beats."
        )
