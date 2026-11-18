"""Tests for SARIMAX training helpers."""
import inspect
from datetime import date, timedelta
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import src.training.train_sarimax as ts
from src.training.train_sarimax import EXOG_COLS, resolve_order, scale_exog
from src.utils.config import SARIMAX_RESEARCH_DAYS


def _make_exog(periods: int = 100) -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=periods, freq="D")
    return pd.DataFrame(
        {
            "temp": [20.0 + i * 0.5 for i in range(periods)],
            "precip": [0.0 if i % 7 != 0 else 1.5 for i in range(periods)],
            "snow_lag1": [0.0] * periods,
            "is_holiday": [0] * periods,
        },
        index=dates,
    )


class TestResolveOrder:
    """The pinned-order cache. If the freshness check is wrong, production
    silently re-searches auto_arima every run — the exact thing the cache exists
    to prevent — and nothing else in the system would report it.
    """

    def _patch(self, monkeypatch, cached, search_result=((2, 1, 2), (1, 0, 1, 7)), winner=0):
        """Stub the cache read/write, the (expensive) auto_arima search, and the
        validation run (``winner`` = index of the candidate it picks)."""
        saved = {}
        monkeypatch.setattr(ts, "_load_cached_order", lambda s3: cached)
        monkeypatch.setattr(ts, "_save_cached_order",
                            lambda s3, o, so, evidence=None: saved.update(order=o, seasonal=so))
        monkeypatch.setattr(ts, "find_best_params", lambda y, x: search_result)
        monkeypatch.setattr(ts, "select_order_by_validation",
                            lambda y, x, cands: (cands[winner], {}))
        return saved

    def test_fresh_cache_is_reused_without_searching(self, monkeypatch):
        cached = ((1, 0, 1), (2, 1, 0, 7), date.today() - timedelta(days=1))
        saved = self._patch(monkeypatch, cached)
        order, seasonal, source = resolve_order(MagicMock(), pd.Series([1.0]), pd.DataFrame())
        assert (order, seasonal) == ((1, 0, 1), (2, 1, 0, 7))
        assert source == "cached"
        assert saved == {}  # nothing re-pinned

    def test_cache_miss_searches_and_pins(self, monkeypatch):
        saved = self._patch(monkeypatch, None)
        order, seasonal, source = resolve_order(MagicMock(), pd.Series([1.0]), pd.DataFrame())
        assert (order, seasonal) == ((2, 1, 2), (1, 0, 1, 7))
        assert source == "auto_arima_stepwise"
        assert saved == {"order": (2, 1, 2), "seasonal": (1, 0, 1, 7)}

    def test_stale_cache_triggers_research(self, monkeypatch):
        stale = date.today() - timedelta(days=SARIMAX_RESEARCH_DAYS + 1)
        saved = self._patch(monkeypatch, ((1, 0, 1), (2, 1, 0, 7), stale))
        order, _, source = resolve_order(MagicMock(), pd.Series([1.0]), pd.DataFrame())
        assert source == "validated"
        assert order == (2, 1, 2)          # the proposal won validation here
        assert saved["order"] == (2, 1, 2)  # and it is re-pinned

    def test_expiring_order_kept_when_it_wins_validation(self, monkeypatch):
        stale = date.today() - timedelta(days=SARIMAX_RESEARCH_DAYS + 1)
        saved = self._patch(monkeypatch, ((1, 0, 1), (2, 1, 0, 7), stale), winner=1)
        order, seasonal, source = resolve_order(MagicMock(), pd.Series([1.0]), pd.DataFrame())
        assert (order, seasonal, source) == ((1, 0, 1), (2, 1, 0, 7), "validated")
        assert saved["order"] == (1, 0, 1)  # re-pinned with a fresh date

    def test_boundary_day_still_counts_as_fresh(self, monkeypatch):
        """Exactly SARIMAX_RESEARCH_DAYS - 1 old is still inside the window."""
        edge = date.today() - timedelta(days=SARIMAX_RESEARCH_DAYS - 1)
        self._patch(monkeypatch, ((1, 0, 1), (2, 1, 0, 7), edge))
        _, _, source = resolve_order(MagicMock(), pd.Series([1.0]), pd.DataFrame())
        assert source == "cached"

    def test_expiry_day_triggers_research(self, monkeypatch):
        """Exactly SARIMAX_RESEARCH_DAYS old is expired (age < N is the rule)."""
        edge = date.today() - timedelta(days=SARIMAX_RESEARCH_DAYS)
        self._patch(monkeypatch, ((1, 0, 1), (2, 1, 0, 7), edge))
        _, _, source = resolve_order(MagicMock(), pd.Series([1.0]), pd.DataFrame())
        assert source == "validated"


class TestSelectOrderByValidation:
    def test_picks_the_order_with_lower_out_of_sample_error(self):
        rng = np.random.default_rng(0)
        idx = pd.date_range("2025-01-01", periods=280, freq="D")
        weekly = np.array([4.0, 4.2, 4.2, 4.2, 3.9, 2.7, 2.1])
        y = pd.Series(weekly[idx.dayofweek] + rng.normal(0, 0.05, len(idx)), index=idx)
        exog = _make_exog(len(idx)).set_axis(idx)
        good, bad = ((1, 0, 0), (1, 1, 0, 7)), ((0, 0, 0), (0, 0, 0, 7))
        best, scores = ts.select_order_by_validation(y, exog, [bad, good])
        assert best == good
        assert set(scores) == {"(1, 0, 0)x(1, 1, 0, 7)", "(0, 0, 0)x(0, 0, 0, 7)"}


class TestFindBestParams:
    def test_exog_reaches_auto_arima(self, monkeypatch):
        """pmdarima silently ignores an unknown keyword like `exog=`, so the order
        search would run without regressors and nothing would fail."""
        assert "X" in inspect.signature(ts.pm.auto_arima).parameters
        captured = {}

        def fake_auto_arima(y, **kwargs):
            captured.update(kwargs)
            return MagicMock(order=(1, 0, 1), seasonal_order=(2, 1, 0, 7))

        monkeypatch.setattr(ts.pm, "auto_arima", fake_auto_arima)
        exog = _make_exog(50)
        ts.find_best_params(pd.Series(range(50), index=exog.index, dtype=float), exog)
        assert captured.get("X") is exog
        assert "exog" not in captured


class TestScaleExog:
    def test_returns_three_values(self):
        exog = _make_exog(100)
        train_idx = exog.index[:70]
        test_idx = exog.index[70:]
        result = scale_exog(exog, train_idx, test_idx)
        assert len(result) == 3

    def test_train_values_in_unit_range(self):
        exog = _make_exog(100)
        train_idx = exog.index[:70]
        test_idx = exog.index[70:]
        train_exog, _, _ = scale_exog(exog, train_idx, test_idx)
        assert train_exog.min().min() >= -1e-9
        assert train_exog.max().max() <= 1.0 + 1e-9

    def test_column_names_preserved(self):
        exog = _make_exog(100)
        train_idx = exog.index[:70]
        test_idx = exog.index[70:]
        train_exog, test_exog, _ = scale_exog(exog, train_idx, test_idx)
        assert list(train_exog.columns) == EXOG_COLS
        assert list(test_exog.columns) == EXOG_COLS

    def test_train_index_preserved(self):
        exog = _make_exog(100)
        train_idx = exog.index[:70]
        test_idx = exog.index[70:]
        train_exog, _, _ = scale_exog(exog, train_idx, test_idx)
        assert list(train_exog.index) == list(train_idx)

    def test_test_index_preserved(self):
        exog = _make_exog(100)
        train_idx = exog.index[:70]
        test_idx = exog.index[70:]
        _, test_exog, _ = scale_exog(exog, train_idx, test_idx)
        assert list(test_exog.index) == list(test_idx)

    def test_returns_fitted_scaler(self):
        exog = _make_exog(100)
        train_idx = exog.index[:70]
        test_idx = exog.index[70:]
        _, _, scaler = scale_exog(exog, train_idx, test_idx)
        assert isinstance(scaler, MinMaxScaler)

    def test_train_and_test_shape_matches_index_lengths(self):
        exog = _make_exog(100)
        train_idx = exog.index[:70]
        test_idx = exog.index[70:]
        train_exog, test_exog, _ = scale_exog(exog, train_idx, test_idx)
        assert len(train_exog) == 70
        assert len(test_exog) == 30
