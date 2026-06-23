"""Tests for the walk-forward evaluator's pure statistical core (no model fitting)."""
import numpy as np
import pytest

from src.evaluation.walk_forward import (
    block_bootstrap_mae_diff,
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
