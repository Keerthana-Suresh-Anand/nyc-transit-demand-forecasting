"""Tests for the walk-forward evaluator's pure statistical core (no model fitting)."""
from datetime import date

import numpy as np
import pytest

import src.evaluation.walk_forward as wf
from src.evaluation.walk_forward import (
    block_bootstrap_mae_diff,
    conformal_half_widths,
    significance_verdict,
    summarize,
)


class TestSignificanceVerdict:
    def test_p1_better_when_ci_entirely_negative(self):
        assert significance_verdict(-0.5, -0.1, "A", "B") == "A significantly better"

    def test_p2_better_when_ci_entirely_positive(self):
        assert significance_verdict(0.1, 0.5, "A", "B") == "B significantly better"

    def test_tie_when_ci_spans_zero(self):
        assert significance_verdict(-0.2, 0.3, "A", "B") == "TIE (95% CI includes 0)"


class TestBlockBootstrap:
    def test_identical_predictions_yield_zero_diff(self):
        blocks = [np.array([0.1, 0.2, 0.3]) for _ in range(4)]
        actual = [np.array([0.15, 0.25, 0.35]) for _ in range(4)]
        mean_d, lo, hi = block_bootstrap_mae_diff(blocks, blocks, actual, n_boot=200)
        assert mean_d == pytest.approx(0.0)
        assert lo == pytest.approx(0.0)
        assert hi == pytest.approx(0.0)

    def test_perfect_p1_beats_offset_p2(self):
        actual = [np.array([3.0, 3.1, 3.2]) for _ in range(4)]
        p1 = [a.copy() for a in actual]            # perfect
        p2 = [a + 0.5 for a in actual]             # constant 0.5 error
        mean_d, lo, hi = block_bootstrap_mae_diff(p1, p2, actual, n_boot=200)
        assert mean_d == pytest.approx(-0.5)
        assert hi < 0  # p1 reliably better


def _blocks(sar, xgb, act):
    return {
        "sarimax": sar, "xgboost": xgb, "actual": act,
        "seasonal_naive": act, "persistence": act, "n_origins": len(act),
    }


class TestSummarize:
    def _data(self):
        act = [np.array([3.0, 3.1, 3.2]), np.array([3.0, 2.9, 3.1]),
               np.array([3.2, 3.3, 3.1]), np.array([2.8, 2.9, 3.0])]
        sar = [a.copy() for a in act]          # SARIMAX perfect
        xgb = [a + 0.5 for a in act]           # XGBoost off by 0.5
        return _blocks(sar, xgb, act)

    def test_mae_table(self):
        r = summarize(self._data(), n_boot=200)
        assert r["mae"]["sarimax"] == pytest.approx(0.0)
        assert r["mae"]["xgboost"] == pytest.approx(0.5)
        assert r["mae"]["ensemble_50_50"] == pytest.approx(0.25)

    def test_best_weight_favors_perfect_model(self):
        r = summarize(self._data(), n_boot=200)
        assert r["best_weight"] == pytest.approx(1.0)
        assert r["best_weight_mae"] == pytest.approx(0.0)

    def test_point_and_origin_counts(self):
        r = summarize(self._data(), n_boot=200)
        assert r["n_origins"] == 4
        assert r["n_points"] == 12

    def test_significance_flags_sarimax_better(self):
        r = summarize(self._data(), n_boot=200)
        sig = r["significance"]["sarimax_vs_xgboost"]
        assert sig["ci_hi"] < 0
        assert sig["verdict"] == "SARIMAX significantly better"


class TestConformalHalfWidths:
    def _blocks(self, resid_by_origin):
        """Build pred/actual blocks with the given signed residuals per origin."""
        actual = [np.array([3.0, 3.1, 3.2]) for _ in resid_by_origin]
        pred = [a + np.asarray(r) for a, r in zip(actual, resid_by_origin, strict=True)]
        return pred, actual

    def test_band_widens_with_horizon_when_error_grows(self):
        # Error grows with lead time: 0.1 / 0.2 / 0.4
        pred, actual = self._blocks([[0.1, 0.2, 0.4]] * 8)
        out = conformal_half_widths(pred, actual, 0.9)
        hw = out["half_width_by_horizon"]
        assert hw[0] < hw[1] < hw[2]

    def test_constant_error_gives_flat_band(self):
        pred, actual = self._blocks([[0.2, 0.2, 0.2]] * 8)
        hw = conformal_half_widths(pred, actual, 0.9)["half_width_by_horizon"]
        assert hw[0] == pytest.approx(hw[1]) == pytest.approx(hw[2])

    def test_higher_coverage_gives_wider_band(self):
        resid = [[0.1 * (i % 5), 0.2, 0.3] for i in range(10)]
        pred, actual = self._blocks(resid)
        lo = conformal_half_widths(pred, actual, 0.80)["half_width_by_horizon"]
        hi = conformal_half_widths(pred, actual, 0.95)["half_width_by_horizon"]
        assert all(wide >= narrow for wide, narrow in zip(hi, lo, strict=True))

    def test_empirical_coverage_meets_nominal_in_sample(self):
        rng = np.random.default_rng(0)
        actual = [np.full(3, 3.0) for _ in range(40)]
        pred = [a + rng.normal(0, 0.2, 3) for a in actual]
        hw = np.asarray(conformal_half_widths(pred, actual, 0.9)["half_width_by_horizon"])
        # Conformal guarantees marginal coverage >= nominal on its calibration set
        covered = sum(
            int(abs(p[h] - a[h]) <= hw[h])
            for p, a in zip(pred, actual, strict=True) for h in range(3)
        )
        assert covered / (40 * 3) >= 0.9

    def test_sign_of_residual_does_not_matter(self):
        pos, actual = self._blocks([[0.2, 0.3, 0.4]] * 6)
        neg, _ = self._blocks([[-0.2, -0.3, -0.4]] * 6)
        a = conformal_half_widths(pos, actual, 0.9)["half_width_by_horizon"]
        b = conformal_half_widths(neg, actual, 0.9)["half_width_by_horizon"]
        assert a == pytest.approx(b)

    def test_metadata_reports_score_count_and_coverage(self):
        pred, actual = self._blocks([[0.1, 0.2, 0.3]] * 7)
        out = conformal_half_widths(pred, actual, 0.9)
        assert out["n_scores"] == 21  # 7 origins x 3 horizon steps
        assert out["coverage"] == 0.9

    def test_summarize_includes_conformal_block(self):
        act = [np.array([3.0, 3.1, 3.2]) for _ in range(4)]
        sar = [a + 0.1 for a in act]
        xgbp = [a - 0.1 for a in act]
        r = summarize(_blocks(sar, xgbp, act), n_boot=100)
        assert len(r["conformal"]["half_width_by_horizon"]) == 3


class TestPinnedProductionOrder:
    """The backtest pins to production's cached SARIMAX order, and degrades to a
    fresh search (returns None) on any cache/credential failure."""

    def test_returns_order_and_seasonal_from_cache(self, monkeypatch):
        monkeypatch.setattr(wf, "get_s3_client", lambda: object())
        monkeypatch.setattr(
            wf, "_load_cached_order",
            lambda s3: ((1, 0, 1), (2, 1, 0, 7), date(2026, 6, 24)),
        )
        assert wf._pinned_production_order() == ((1, 0, 1), (2, 1, 0, 7))

    def test_returns_none_on_cache_miss(self, monkeypatch):
        monkeypatch.setattr(wf, "get_s3_client", lambda: object())
        monkeypatch.setattr(wf, "_load_cached_order", lambda s3: None)
        assert wf._pinned_production_order() is None

    def test_returns_none_when_s3_unavailable(self, monkeypatch):
        def boom():
            raise RuntimeError("no creds")

        monkeypatch.setattr(wf, "get_s3_client", boom)
        assert wf._pinned_production_order() is None
