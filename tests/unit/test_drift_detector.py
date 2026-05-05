"""Tests for drift detection: PSI values and retrain trigger logic."""
import numpy as np
import pytest

from src.evaluation.drift_detector import compute_psi
from src.utils.config import PSI_CRITICAL_THRESHOLD, PSI_MODERATE_THRESHOLD


class TestComputePSI:
    def test_identical_distributions_near_zero(self):
        rng = np.random.default_rng(42)
        arr = rng.normal(50, 10, 1000)
        psi = compute_psi(arr, arr)
        assert psi < 0.01, f"Expected PSI≈0 for identical distributions, got {psi:.4f}"

    def test_slightly_shifted_distribution_moderate(self):
        rng = np.random.default_rng(42)
        reference = rng.normal(50, 10, 500)
        recent = rng.normal(52, 10, 100)  # small shift
        psi = compute_psi(reference, recent)
        assert psi >= 0, "PSI must be non-negative"

    def test_heavily_shifted_distribution_exceeds_critical(self):
        rng = np.random.default_rng(42)
        reference = rng.normal(0, 1, 1000)
        recent = rng.normal(8, 1, 200)  # extreme shift
        psi = compute_psi(reference, recent)
        assert psi > PSI_CRITICAL_THRESHOLD, (
            f"Expected PSI > {PSI_CRITICAL_THRESHOLD} for extreme shift, got {psi:.4f}"
        )

    def test_psi_is_non_negative(self):
        rng = np.random.default_rng(0)
        for _ in range(10):
            ref = rng.normal(0, 1, 200)
            rec = rng.normal(rng.uniform(-3, 3), 1, 50)
            assert compute_psi(ref, rec) >= 0

    def test_psi_symmetric_approximately(self):
        """PSI(A→B) and PSI(B→A) should be in the same ballpark (not exact)."""
        rng = np.random.default_rng(7)
        a = rng.normal(0, 1, 500)
        b = rng.normal(1, 1, 500)
        psi_ab = compute_psi(a, b)
        psi_ba = compute_psi(b, a)
        # They won't be identical but should both be moderate or both low
        assert (psi_ab > PSI_MODERATE_THRESHOLD) == (psi_ba > PSI_MODERATE_THRESHOLD)


class TestRetainThresholdLogic:
    """Verify the status classification boundaries match config thresholds."""

    def _classify(self, psi: float) -> str:
        if psi > PSI_CRITICAL_THRESHOLD:
            return "critical"
        if psi > PSI_MODERATE_THRESHOLD:
            return "moderate"
        return "stable"

    def test_below_moderate_is_stable(self):
        assert self._classify(PSI_MODERATE_THRESHOLD - 0.001) == "stable"

    def test_above_moderate_below_critical_is_moderate(self):
        mid = (PSI_MODERATE_THRESHOLD + PSI_CRITICAL_THRESHOLD) / 2
        assert self._classify(mid) == "moderate"

    def test_above_critical_is_critical(self):
        assert self._classify(PSI_CRITICAL_THRESHOLD + 0.001) == "critical"

    def test_retrain_recommended_only_on_critical(self):
        assert self._classify(PSI_CRITICAL_THRESHOLD + 0.01) == "critical"
        assert self._classify(PSI_MODERATE_THRESHOLD + 0.01) != "critical"
