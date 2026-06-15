"""Tests for monitor_performance: forecast metrics, PSI scores, and run() orchestration."""
from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.monitoring.monitor_performance import (
    _compute_forecast_metrics,
    _compute_psi_scores,
    run,
)
from src.utils.config import PSI_CRITICAL_THRESHOLD, PSI_MODERATE_THRESHOLD


def _make_gold(periods=120) -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=periods, freq="D")
    return pd.DataFrame(
        {
            "daily_ridership": [3_000_000 + i * 5_000 for i in range(periods)],
            "temp": [40.0 + i * 0.2 for i in range(periods)],
            "precip": [0.0] * periods,
            "snow": [0.0] * periods,
        },
        index=dates,
    )


def _make_past_forecasts(gold: pd.DataFrame, n_rows: int = 10) -> pd.DataFrame:
    past_dates = gold.index[-n_rows:].date
    return pd.DataFrame(
        {
            "date": past_dates,
            "ensemble_forecast_M": [3.0] * n_rows,
        }
    )


class TestComputeForecastMetrics:
    def test_returns_zero_evaluated_when_no_date_matches(self):
        gold = _make_gold(30)
        past_fc = pd.DataFrame({
            "date": [date(2020, 1, 1)],  # not in gold
            "ensemble_forecast_M": [3.0],
        })
        result = _compute_forecast_metrics(past_fc, gold)
        assert result["n_evaluated"] == 0

    def test_computes_mae_correctly(self):
        gold = _make_gold(30)
        match_date = gold.index[-1]
        actual_M = gold.loc[match_date, "daily_ridership"] / 1_000_000
        predicted_M = actual_M + 0.5  # error of 0.5M

        past_fc = pd.DataFrame({
            "date": [match_date.date()],
            "ensemble_forecast_M": [predicted_M],
        })
        result = _compute_forecast_metrics(past_fc, gold)
        assert result["n_evaluated"] == 1
        assert result["rolling_mae_M"] == pytest.approx(0.5, abs=1e-6)

    def test_mae_is_symmetric(self):
        gold = _make_gold(30)
        match_date = gold.index[-1]
        actual_M = gold.loc[match_date, "daily_ridership"] / 1_000_000

        over = pd.DataFrame({"date": [match_date.date()], "ensemble_forecast_M": [actual_M + 1.0]})
        under = pd.DataFrame({"date": [match_date.date()], "ensemble_forecast_M": [actual_M - 1.0]})
        assert _compute_forecast_metrics(over, gold)["rolling_mae_M"] == pytest.approx(
            _compute_forecast_metrics(under, gold)["rolling_mae_M"]
        )

    def test_returns_mape_as_percentage(self):
        gold = _make_gold(30)
        match_date = gold.index[-1]
        actual_M = gold.loc[match_date, "daily_ridership"] / 1_000_000

        past_fc = pd.DataFrame({
            "date": [match_date.date()],
            "ensemble_forecast_M": [actual_M * 1.10],  # 10% over
        })
        result = _compute_forecast_metrics(past_fc, gold)
        assert result["rolling_mape_pct"] == pytest.approx(10.0, abs=0.1)


class TestComputePsiScores:
    def test_returns_empty_when_insufficient_data(self):
        # fewer than REFERENCE_DAYS + RECENT_DAYS = 90 + 14 = 104 rows
        gold = _make_gold(50)
        result = _compute_psi_scores(gold)
        assert result == {}

    def test_returns_dict_keyed_by_feature_columns(self):
        gold = _make_gold(120)
        result = _compute_psi_scores(gold)
        for col in ["temp", "precip", "snow"]:
            assert col in result

    def test_psi_values_are_non_negative(self):
        gold = _make_gold(120)
        result = _compute_psi_scores(gold)
        for col, psi in result.items():
            assert psi >= 0, f"PSI for {col} should be non-negative, got {psi}"

    def test_stable_distribution_yields_low_psi(self):
        # Identical distribution across reference and recent windows → PSI near 0
        periods = 120
        dates = pd.date_range("2025-01-01", periods=periods, freq="D")
        gold = pd.DataFrame(
            {"daily_ridership": [3_000_000] * periods, "temp": [50.0] * periods,
             "precip": [0.0] * periods, "snow": [0.0] * periods},
            index=dates,
        )
        result = _compute_psi_scores(gold)
        for psi in result.values():
            assert psi < PSI_MODERATE_THRESHOLD


class TestMonitorRun:
    def test_returns_error_status_when_no_gold_data(self, monkeypatch):
        monkeypatch.setattr("src.monitoring.monitor_performance.get_s3_client", MagicMock())
        with patch("src.monitoring.monitor_performance._load_gold", return_value=None):
            result = run()
        assert result["status"] == "error"

    def test_no_retrain_on_critical_psi_alone(self, monkeypatch):
        """Critical PSI is informational only — must not trigger retraining."""
        gold = _make_gold(120)
        monkeypatch.setattr("src.monitoring.monitor_performance.get_s3_client", MagicMock())

        with patch("src.monitoring.monitor_performance._load_gold", return_value=gold), \
             patch("src.monitoring.monitor_performance._load_past_forecasts", return_value=None), \
             patch("src.monitoring.monitor_performance._compute_psi_scores",
                   return_value={"temp": PSI_CRITICAL_THRESHOLD + 0.1}), \
             patch("src.monitoring.monitor_performance.write_s3_json"):
            result = run()

        assert result["retrain_recommended"] is False
        assert result["psi_status"] == "critical"

    def test_no_retrain_when_stable(self, monkeypatch):
        gold = _make_gold(120)
        monkeypatch.setattr("src.monitoring.monitor_performance.get_s3_client", MagicMock())

        with patch("src.monitoring.monitor_performance._load_gold", return_value=gold), \
             patch("src.monitoring.monitor_performance._load_past_forecasts", return_value=None), \
             patch("src.monitoring.monitor_performance._compute_psi_scores",
                   return_value={"temp": 0.01, "precip": 0.01, "snow": 0.0}), \
             patch("src.monitoring.monitor_performance.write_s3_json"):
            result = run()

        assert result["retrain_recommended"] is False

    def test_retrain_recommended_when_mae_exceeds_threshold(self, monkeypatch):
        gold = _make_gold(120)
        past_fc = _make_past_forecasts(gold, n_rows=5)
        monkeypatch.setattr("src.monitoring.monitor_performance.get_s3_client", MagicMock())

        with patch("src.monitoring.monitor_performance._load_gold", return_value=gold), \
             patch("src.monitoring.monitor_performance._load_past_forecasts", return_value=past_fc), \
             patch("src.monitoring.monitor_performance._compute_psi_scores",
                   return_value={"temp": 0.01}), \
             patch("src.monitoring.monitor_performance._compute_forecast_metrics",
                   return_value={"n_evaluated": 5, "rolling_mae_M": 1.0, "rolling_mape_pct": 30.0}), \
             patch("src.monitoring.monitor_performance.write_s3_json"):
            # training_mae=0.5; rolling=1.0 > 1.5 * 0.5 = 0.75 → retrain
            result = run(training_mae=0.5)

        assert result["retrain_recommended"] is True

    def test_report_includes_psi_scores(self, monkeypatch):
        gold = _make_gold(120)
        monkeypatch.setattr("src.monitoring.monitor_performance.get_s3_client", MagicMock())

        with patch("src.monitoring.monitor_performance._load_gold", return_value=gold), \
             patch("src.monitoring.monitor_performance._load_past_forecasts", return_value=None), \
             patch("src.monitoring.monitor_performance._compute_psi_scores",
                   return_value={"temp": 0.05, "precip": 0.02}), \
             patch("src.monitoring.monitor_performance.write_s3_json"):
            result = run()

        assert "psi_scores" in result
        assert result["psi_scores"]["temp"] == pytest.approx(0.05)
