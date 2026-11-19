"""Tests for the shared rolling-origin helpers.

The leakage tests matter most: every holdout MAE, gate decision and backtest number
is computed through these functions, so a forecast that reads past its origin
would silently inflate all of them.
"""
import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.evaluation.origins import (
    forecast_origins,
    origin_records,
    persistence_path,
    sarimax_origin_forecasts,
    seasonal_naive_path,
    xgb_origin_forecasts,
)
from src.transformation.preprocess_ml import build_features
from src.utils.features import cast_categoricals

H = 14


def _gold(n=200) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=n, freq="D")
    rng = np.random.default_rng(1)
    weekly = np.array([4.0, 4.2, 4.2, 4.2, 3.9, 2.7, 2.1])
    return pd.DataFrame({
        "daily_ridership": (weekly[idx.dayofweek] + rng.normal(0, 0.05, n)) * 1_000_000,
        "temp": 10 + rng.normal(0, 3, n), "precip": np.abs(rng.normal(0, 2, n)),
        "snow": np.zeros(n), "snow_lag1": np.zeros(n), "is_holiday": np.zeros(n, dtype=int),
    }, index=idx)


class TestNaivePaths:
    def test_seasonal_naive_repeats_the_last_known_week(self):
        history = np.arange(21, dtype=float)  # origin value = 20
        path = seasonal_naive_path(history, H)
        assert list(path[:7]) == list(history[-7:])
        assert list(path[7:]) == list(history[-7:])
        assert path.max() <= history[-1]

    def test_persistence_holds_the_origin_value(self):
        assert (persistence_path(np.array([1.0, 2.0, 5.0]), H) == 5.0).all()


class TestOriginRecords:
    def test_long_frame_aligns_dates_steps_and_actuals(self):
        y = pd.Series(np.arange(60, dtype=float), index=pd.date_range("2025-01-01", periods=60))
        origins = [y.index[20], y.index[27]]
        recs = origin_records(origins, [np.zeros(H), np.ones(H)], y)
        assert len(recs) == 2 * H
        first = recs[recs["origin"] == origins[0]]
        assert first["date"].iloc[0] == origins[0] + pd.Timedelta(days=1)
        assert list(first["step"]) == list(range(1, H + 1))
        assert (first["y_true"].to_numpy() == y.iloc[21:35].to_numpy()).all()


class TestNoLookAhead:
    def test_sarimax_forecast_ignores_actuals_after_origin(self):
        gold = _gold()
        y = gold["daily_ridership"] / 1e6
        exog = gold[["temp", "precip", "snow_lag1", "is_holiday"]]
        train_end = 150
        res = SARIMAX(y.iloc[:train_end], exog=exog.iloc[:train_end], order=(1, 0, 0),
                      seasonal_order=(1, 0, 0, 7)).fit(disp=False)
        origins = forecast_origins(len(y), 45, H)
        base = sarimax_origin_forecasts(res, y, exog, train_end, origins, H)

        y_bumped = y.copy()
        y_bumped.iloc[origins[0] + 1:] += 10.0  # change every actual after the 1st origin
        bumped = sarimax_origin_forecasts(res, y_bumped, exog, train_end, origins, H)
        np.testing.assert_allclose(base[0], bumped[0])
        assert not np.allclose(base[-1], bumped[-1])  # later origins do see the new actuals

    def test_xgboost_forecast_ignores_actuals_after_origin(self):
        df_ml = cast_categoricals(build_features(_gold()))
        feats = [c for c in df_ml.columns if c != "daily_ridership"]
        model = xgb.XGBRegressor(n_estimators=30, max_depth=3, enable_categorical=True,
                                 tree_method="hist").fit(df_ml[feats].iloc[:120],
                                                         df_ml["daily_ridership"].iloc[:120] / 1e6)
        origin = df_ml.index[140]
        base = xgb_origin_forecasts(model, df_ml, [origin], H)[0]
        bumped_df = df_ml.copy()
        bumped_df.loc[bumped_df.index > origin, "daily_ridership"] *= 3
        bumped = xgb_origin_forecasts(model, bumped_df, [origin], H)[0]
        np.testing.assert_allclose(base, bumped)


class TestForecastOrigins:
    @pytest.mark.parametrize("n", [100, 101, 102, 103, 104, 105, 106])
    def test_origins_share_the_last_actuals_weekday(self, n):
        assert all((n - 1 - p) % 7 == 0 for p in forecast_origins(n, 30, H))

    def test_holdout_window_yields_three_full_origins(self):
        n = 500
        origins = forecast_origins(n, 30, H)
        assert len(origins) == 3
        assert min(origins) >= n - 30 - 1 and max(origins) + H <= n - 1
