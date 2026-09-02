"""Tests for forecast generation: output shape, date range, autoregressive XGBoost loop,
SARIMAX state re-anchoring."""
import contextlib
from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.prediction.generate_forecast import FORECAST_DAYS, _reanchor_sarimax, xgboost_forecast
from src.utils.config import SARIMAX_EXOG_COLS


@contextlib.contextmanager
def _patched_model_loading(mock_model):
    """Patch registry resolution + model loading so xgboost_forecast runs offline."""
    with patch("src.prediction.generate_forecast.mlflow") as mock_mlflow, \
         patch("src.prediction.generate_forecast.MlflowClient"), \
         patch("src.prediction.generate_forecast.production_model_uri",
               return_value="models:/xgboost_production/1"):
        mock_mlflow.xgboost.load_model.return_value = mock_model
        yield


def _make_ml_df(periods=30) -> pd.DataFrame:
    """Build a minimal ML gold DataFrame matching the schema expected by xgboost_forecast."""
    dates = pd.date_range(end=date.today() - timedelta(days=1), periods=periods, freq="D")
    data = {
        "daily_ridership": [3_000_000.0 + i * 5_000 for i in range(periods)],
        "day_of_week": [d.dayofweek for d in dates],
        "month": [d.month for d in dates],
        "is_weekend": [int(d.dayofweek >= 5) for d in dates],
        "is_holiday": [0] * periods,
        "temp": [55.0] * periods,
        "precip": [0.0] * periods,
        "snow": [0.0] * periods,
        "snow_lag1": [0.0] * periods,
        "temp_lag1": [54.0] * periods,
        "precip_lag1": [0.0] * periods,
        "ridership_lag1": [3.0] * periods,
        "ridership_lag2": [3.0] * periods,
        "ridership_lag3": [3.0] * periods,
        "ridership_lag7": [3.0] * periods,
        "ridership_lag14": [3.0] * periods,
        "ridership_14d_avg": [3.0] * periods,
        "ridership_7d_std": [0.05] * periods,
    }
    return pd.DataFrame(data, index=dates)


def _make_weather_fcst(days=14) -> pd.DataFrame:
    future = [date.today() + timedelta(days=i + 1) for i in range(days)]
    return pd.DataFrame({
        "datetime": future,
        "temp": [60.0] * days,
        "precip": [0.0] * days,
        "snow": [0.0] * days,
    })


class TestXGBoostForecast:
    def _mock_model(self, return_value: float = 3.1):
        m = MagicMock()
        m.predict.return_value = np.array([return_value])
        return m

    def test_output_has_exactly_14_rows(self):
        df_ml = _make_ml_df()
        weather = _make_weather_fcst()
        mock_model = self._mock_model()
        with _patched_model_loading(mock_model):
            result = xgboost_forecast(df_ml, weather, date.today() + timedelta(days=1))
        assert len(result) == FORECAST_DAYS

    def test_first_forecast_date_is_tomorrow(self):
        """The autoregressive loop must start at today+1."""
        df_ml = _make_ml_df()
        weather = _make_weather_fcst()
        call_dates: list[date] = []

        def capture_predict(X):
            # infer the pred_date from the day_of_week feature
            call_dates.append(X.iloc[0]["day_of_week"])
            return np.array([3.1])

        mock_model = MagicMock()
        mock_model.predict.side_effect = capture_predict

        with _patched_model_loading(mock_model):
            xgboost_forecast(df_ml, weather, date.today() + timedelta(days=1))

        expected_first_dow = (date.today() + timedelta(days=1)).weekday()
        assert call_dates[0] == expected_first_dow

    def test_autoregressive_loop_uses_predicted_not_actual(self):
        """lag1 for step 2 must come from step 1's prediction, not from historical data."""
        df_ml = _make_ml_df()
        weather = _make_weather_fcst()
        predictions_seen: list[float] = []

        step1_pred = 9.99  # distinctive sentinel value

        def stepped_predict(X):
            lag1 = float(X.iloc[0]["ridership_lag1"])
            predictions_seen.append(lag1)
            if len(predictions_seen) == 1:
                return np.array([step1_pred])
            return np.array([3.1])

        mock_model = MagicMock()
        mock_model.predict.side_effect = stepped_predict

        with _patched_model_loading(mock_model):
            xgboost_forecast(df_ml, weather, date.today() + timedelta(days=1))

        # Step 2's lag1 must equal step 1's prediction
        assert predictions_seen[1] == pytest.approx(step1_pred)

    def test_ridership_lags_slide_forward_across_horizon(self):
        """Regression: ridership lags must advance one day per forecast step, not stay
        frozen at the last-known window. A frozen lag-14 (identical value fed to all 14
        days) was a train/serve skew bug — the holdout/backtest slide these features, so
        production must too."""
        df_ml = _make_ml_df(periods=30)
        weather = _make_weather_fcst()
        lag14_seen: list[float] = []

        def capture(X):
            lag14_seen.append(float(X.iloc[0]["ridership_lag14"]))
            return np.array([3.1])

        mock_model = MagicMock()
        mock_model.predict.side_effect = capture
        with _patched_model_loading(mock_model):
            xgboost_forecast(df_ml, weather, date.today() + timedelta(days=1))

        assert len(lag14_seen) == FORECAST_DAYS
        assert lag14_seen == sorted(lag14_seen)        # slides forward (monotone)
        assert len(set(lag14_seen)) == FORECAST_DAYS   # all distinct — not frozen
        # Step 0's lag-14 is the actual ridership 14 days before the first forecast day.
        expected_first = float(df_ml["daily_ridership"].iloc[-14] / 1_000_000)
        assert lag14_seen[0] == pytest.approx(expected_first)

    def test_output_is_numpy_array_of_floats(self):
        df_ml = _make_ml_df()
        weather = _make_weather_fcst()
        with _patched_model_loading(self._mock_model()):
            result = xgboost_forecast(df_ml, weather, date.today() + timedelta(days=1))
        assert isinstance(result, np.ndarray)
        assert result.dtype.kind == "f"


class TestConformalBand:
    """The served band must come from the walk-forward's ensemble residuals, and
    degrade visibly (never silently) when that calibration is unavailable."""

    def _band(self, wf_json, pred=None):
        from src.prediction.generate_forecast import _conformal_band

        pred = np.full(3, 3.0) if pred is None else pred
        s3 = MagicMock()
        with patch("src.prediction.generate_forecast.read_s3_json", return_value=wf_json):
            return _conformal_band(s3, pred), pred

    def test_uses_conformal_half_widths_per_horizon(self):
        wf = {"run_date": "2026-09-01",
              "conformal": {"half_width_by_horizon": [0.1, 0.2, 0.3], "n_scores": 154}}
        (lo, hi, meta), pred = self._band(wf)
        assert meta["method"] == "conformal_walkforward"
        assert lo == pytest.approx(pred - np.array([0.1, 0.2, 0.3]))
        assert hi == pytest.approx(pred + np.array([0.1, 0.2, 0.3]))

    def test_band_is_centered_on_the_ensemble(self):
        wf = {"conformal": {"half_width_by_horizon": [0.1, 0.25, 0.4]}}
        (lo, hi, _), pred = self._band(wf)
        assert (lo + hi) / 2 == pytest.approx(pred)

    def test_falls_back_to_flat_mae_band_without_conformal_block(self):
        (lo, hi, meta), pred = self._band({"mae": {"ensemble_50_50": 0.2}})
        assert meta["method"] == "fallback_flat_mae"
        width = hi - lo
        assert width[0] == pytest.approx(width[-1])  # flat, not horizon-scaled

    def test_marks_unavailable_when_nothing_to_calibrate_on(self):
        (lo, hi, meta), _ = self._band({})
        assert meta["method"] == "unavailable"
        assert np.isnan(lo).all() and np.isnan(hi).all()

    def test_shorter_calibration_than_horizon_falls_back(self):
        # 2 half-widths can't cover a 3-day horizon — must not silently truncate
        wf = {"conformal": {"half_width_by_horizon": [0.1, 0.2]}, "mae": {"ensemble_50_50": 0.2}}
        (_, _, meta), _ = self._band(wf)
        assert meta["method"] == "fallback_flat_mae"

    def test_s3_failure_degrades_instead_of_raising(self):
        from src.prediction.generate_forecast import _conformal_band

        with patch("src.prediction.generate_forecast.read_s3_json",
                   side_effect=RuntimeError("no such key")):
            lo, hi, meta = _conformal_band(MagicMock(), np.full(3, 3.0))
        assert meta["method"] == "unavailable"
        assert np.isnan(lo).all()


def _make_sarima_df(periods: int) -> pd.DataFrame:
    """Minimal gold-SARIMA frame: daily ridership + the exog columns, daily freq."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2025-01-01", periods=periods, freq="D")
    df = pd.DataFrame({
        "daily_ridership": 3_000_000 + rng.normal(0, 50_000, periods).cumsum(),
        "temp": 40 + 10 * np.sin(np.arange(periods) / 14),
        "precip": rng.uniform(0, 0.3, periods),
        "snow_lag1": 0.0,
        "is_holiday": 0,
    }, index=dates)
    return df.asfreq("D")


def _fit_sarimax(df: pd.DataFrame, scaler):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    y = df["daily_ridership"] / 1_000_000
    exog = pd.DataFrame(
        scaler.transform(df[SARIMAX_EXOG_COLS]), index=df.index, columns=SARIMAX_EXOG_COLS
    )
    return SARIMAX(y, exog=exog, order=(1, 0, 0),
                   enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)


class TestReanchorSarimax:
    """The registered model's state must be advanced over actuals that arrived after
    training, so the 14-day forecast starts at the latest actual — not the training end."""

    def _setup(self, total_days=74, train_days=60):
        from sklearn.preprocessing import MinMaxScaler
        df = _make_sarima_df(total_days)
        scaler = MinMaxScaler().fit(df[SARIMAX_EXOG_COLS])
        model = _fit_sarimax(df.iloc[:train_days], scaler)
        return df, scaler, model

    def test_appends_new_actuals_and_moves_anchor(self):
        df, scaler, model = self._setup()
        assert model.fittedvalues.index[-1] == df.index[59]  # stale anchor before fix

        res = _reanchor_sarimax(model, df, scaler)
        assert res.fittedvalues.index[-1] == df.index[-1]
        assert res.nobs == len(df)

    def test_forecast_starts_after_latest_actual(self):
        df, scaler, model = self._setup()
        res = _reanchor_sarimax(model, df, scaler)

        future_exog = pd.DataFrame(
            np.zeros((3, len(SARIMAX_EXOG_COLS))),
            index=pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=3, freq="D"),
            columns=SARIMAX_EXOG_COLS,
        )
        fc = res.get_forecast(steps=3, exog=future_exog)
        assert fc.predicted_mean.index[0] == df.index[-1] + pd.Timedelta(days=1)

    def test_noop_when_no_new_actuals(self):
        df, scaler, model = self._setup(total_days=60, train_days=60)
        res = _reanchor_sarimax(model, df, scaler)
        assert res is model  # returned unchanged, nothing appended

    def test_parameters_unchanged_by_reanchor(self):
        """refit=False must advance state without re-estimating coefficients."""
        df, scaler, model = self._setup()
        params_before = model.params.copy()
        res = _reanchor_sarimax(model, df, scaler)
        pd.testing.assert_series_equal(params_before, res.params)
