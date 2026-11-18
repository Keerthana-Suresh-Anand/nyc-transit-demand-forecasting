"""Rolling-origin forecasting, shared by the training holdouts, the promotion gate
and the walk-forward backtest.

An origin is the last day whose actual is known when a forecast is made; the
forecast covers the `horizon` days after it. Production forecasts 14 days from the
latest actual and re-anchors weekly, so every evaluation here does the same: weekly
origins anchored to the series' last day (so they share production's origin
weekday), each forecast made only from information known at its origin.

Keeping one implementation means a holdout MAE, a gate decision and the backtest
all measure the same task — a single long forecast instead would compound the
recursive XGBoost error over horizons production never serves.
"""
import math

import numpy as np
import pandas as pd

from src.utils.features import iterative_xgb_predict

ORIGIN_STEP = 7  # weekly re-anchor, matching production


def forecast_origins(n: int, eval_days: int, horizon: int, step: int = ORIGIN_STEP) -> list[int]:
    """Origin positions within the last `eval_days` of an n-day series, each followed
    by a full `horizon` of actuals, oldest first.

    Anchored to the series' last day and stepping back whole weeks, so every origin
    falls on the same weekday as production's origin (it forecasts from the last
    actual). A forward step from the window start would put origins on an arbitrary
    weekday — and since horizon step k then always lands on one weekday, per-step
    statistics (e.g. the conformal band's widths) would describe the wrong weekdays.
    """
    last_origin = n - 1 - 7 * math.ceil(horizon / 7)
    first_allowed = n - eval_days - 1
    return list(range(last_origin, first_allowed - 1, -step))[::-1]


def seasonal_naive_path(history: np.ndarray, horizon: int) -> np.ndarray:
    """Seasonal-naive forecast from the end of `history`: step h copies the same
    weekday from the latest week known at the origin (last week for steps 1-7, two
    weeks back for 8-14, ...). Never reads past the origin, so it faces the same
    multi-step task as the models it benchmarks."""
    last = len(history) - 1
    return np.array([history[last + h - 7 * math.ceil(h / 7)] for h in range(1, horizon + 1)],
                    dtype=float)


def persistence_path(history: np.ndarray, horizon: int) -> np.ndarray:
    """The last known value held flat for the whole horizon."""
    return np.full(horizon, float(history[-1]))


def sarimax_origin_forecasts(
    result, y: pd.Series, exog_scaled: pd.DataFrame, train_end: int,
    origins: list[int], horizon: int,
) -> list[np.ndarray]:
    """Forecast `horizon` days from each origin with a SARIMAX fit on positions
    [0:train_end]. The state is advanced over the actuals up to the origin with the
    parameters fixed (`append(refit=False)`), as production re-anchors between
    monthly retrains."""
    out = []
    for p in origins:
        if p >= train_end:
            res = result.append(y.iloc[train_end:p + 1],
                                exog=exog_scaled.iloc[train_end:p + 1], refit=False)
        else:
            res = result
        future = exog_scaled.iloc[p + 1:p + 1 + horizon]
        out.append(np.asarray(res.get_forecast(steps=horizon, exog=future).predicted_mean))
    return out


def xgb_origin_forecasts(
    model, df_ml: pd.DataFrame, origin_dates: list[pd.Timestamp], horizon: int,
) -> list[np.ndarray]:
    """Forecast `horizon` days from each origin date iteratively, seeding lags from
    the actuals known at that origin (the serving feature path)."""
    return [iterative_xgb_predict(model, df_ml, df_ml.index.get_loc(d) + 1, horizon)
            for d in origin_dates]


def origin_records(
    origin_dates: list[pd.Timestamp], preds: list[np.ndarray], y: pd.Series,
) -> pd.DataFrame:
    """Long frame of per-origin forecasts: origin, date, step, y_true, y_pred.
    `y` is indexed by date and in the same units as the predictions."""
    rows = []
    for origin, pred in zip(origin_dates, preds, strict=True):
        dates = pd.date_range(origin + pd.Timedelta(days=1), periods=len(pred), freq="D")
        rows.append(pd.DataFrame({
            "origin": origin, "date": dates, "step": np.arange(1, len(pred) + 1),
            "y_true": y.loc[dates].to_numpy(), "y_pred": pred,
        }))
    return pd.concat(rows, ignore_index=True)
