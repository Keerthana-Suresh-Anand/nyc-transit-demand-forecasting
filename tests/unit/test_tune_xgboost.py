"""Tests for the XGBoost tuning stage: fold layout, the adopt/keep decision and the
persisted-parameter lifecycle. Model fitting is stubbed out."""
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from src.training import tune_xgboost as tx
from src.utils.config import (
    TEST_DAYS,
    XGB_RETUNE_DAYS,
    XGB_TUNING_FOLD_SPACING_DAYS,
    XGB_TUNING_FOLDS,
    XGB_TUNING_VAL_DAYS,
)


def _frame(n: int = 600):
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    X = pd.DataFrame({"f": np.arange(n, dtype=float)}, index=idx)
    return X, X, pd.Series(np.ones(n), index=idx)


class TestFoldEnds:
    def test_last_fold_ends_at_the_data_end_and_folds_are_spaced(self):
        ends = tx.fold_ends(600)
        assert len(ends) == XGB_TUNING_FOLDS
        assert ends[-1] == 600
        assert all(b - a == XGB_TUNING_FOLD_SPACING_DAYS for a, b in zip(ends, ends[1:], strict=False))

    def test_rejects_too_little_data(self):
        too_small = (XGB_TUNING_FOLDS - 1) * XGB_TUNING_FOLD_SPACING_DAYS + XGB_TUNING_VAL_DAYS + 2 * TEST_DAYS
        with pytest.raises(ValueError):
            tx.fold_ends(too_small)

    def test_validation_windows_never_reach_past_the_given_rows(self, monkeypatch):
        seen = []
        monkeypatch.setattr(tx, "fit_recipe", lambda p, X, y: (seen.append(len(X)), None)[::-1])
        monkeypatch.setattr(tx, "xgb_origin_forecasts",
                            lambda m, df, dates, h: [np.zeros(h) for _ in dates])
        X, df, y = _frame()
        tx.fold_mae({}, df, X, y, 600)
        assert seen == [600 - XGB_TUNING_VAL_DAYS]


class TestTuneDecision:
    @pytest.fixture
    def stub_scores(self, monkeypatch):
        """Baseline scores `base`; every trial scores `trial` on every fold."""
        def apply(base: float, trial: float):
            monkeypatch.setattr(tx, "cv_mae", lambda p, df, X, y: base)
            monkeypatch.setattr(tx, "fold_mae", lambda p, df, X, y, end: trial)
        return apply

    def test_challenger_needs_the_minimum_gain_to_replace_serving(self, stub_scores):
        stub_scores(base=0.200, trial=0.199)  # 0.5% better: within noise margin
        serving = {"max_depth": 5}
        res = tx.tune(*_frame(), serving=serving, n_trials=3)
        assert not res.replaced and res.params == serving and res.baseline == "serving"

    def test_clearly_better_challenger_replaces_serving(self, stub_scores):
        stub_scores(base=0.200, trial=0.150)
        res = tx.tune(*_frame(), serving={"max_depth": 5}, n_trials=3)
        assert res.replaced and res.params == res.best_trial_params
        assert res.validation_mae == pytest.approx(0.150)

    def test_first_run_takes_the_better_of_search_and_library_defaults(self, stub_scores):
        stub_scores(base=0.200, trial=0.199)
        res = tx.tune(*_frame(), serving=None, n_trials=3)
        assert res.replaced and res.baseline == "library_defaults"

    def test_first_run_keeps_library_defaults_when_nothing_beats_them(self, stub_scores):
        stub_scores(base=0.100, trial=0.199)
        res = tx.tune(*_frame(), serving=None, n_trials=3)
        assert not res.replaced and res.params == tx.LIBRARY_DEFAULTS

    def test_trials_stay_inside_the_search_space(self, stub_scores):
        stub_scores(base=0.2, trial=0.1)
        res = tx.tune(*_frame(), serving=None, n_trials=5)
        for t in res.trials:
            for name, spec in tx.XGB_SEARCH_SPACE.items():
                assert spec["low"] <= t["params"][name] <= spec["high"]


class TestResolveParams:
    @pytest.fixture
    def fake_tune(self, monkeypatch):
        calls = []

        def tune(df, X, y, serving):
            calls.append(serving)
            return tx.TuningResult(params={"max_depth": 4}, validation_mae=0.2, replaced=True,
                                   baseline="serving", baseline_mae=0.21,
                                   best_trial_params={"max_depth": 4}, best_trial_mae=0.2,
                                   fold_end_dates=[])
        monkeypatch.setattr(tx, "tune", tune)
        saved = []
        monkeypatch.setattr(tx, "_save_cached", lambda s3, r, d: saved.append(r.params))
        return calls, saved

    def _cache(self, monkeypatch, age_days: int | None):
        value = None if age_days is None else (
            {"max_depth": 5}, date.today() - timedelta(days=age_days))
        monkeypatch.setattr(tx, "_load_cached", lambda s3: value)

    def test_fresh_cache_is_reused_without_tuning(self, monkeypatch, fake_tune):
        self._cache(monkeypatch, XGB_RETUNE_DAYS - 1)
        params, source, result = tx.resolve_params(None, *_frame())
        assert (params, source, result) == ({"max_depth": 5}, "cached", None)
        assert fake_tune[0] == []

    def test_expired_cache_is_retuned_against_the_serving_set(self, monkeypatch, fake_tune):
        self._cache(monkeypatch, XGB_RETUNE_DAYS)
        params, source, _ = tx.resolve_params(None, *_frame())
        assert source == "tuned" and params == {"max_depth": 4}
        assert fake_tune[0] == [{"max_depth": 5}] and fake_tune[1] == [{"max_depth": 4}]

    def test_no_cache_tunes_from_scratch(self, monkeypatch, fake_tune):
        self._cache(monkeypatch, None)
        _, source, _ = tx.resolve_params(None, *_frame())
        assert source == "tuned" and fake_tune[0] == [None]

    def test_failed_tuning_keeps_expired_set(self, monkeypatch):
        self._cache(monkeypatch, XGB_RETUNE_DAYS + 5)
        monkeypatch.setattr(tx, "tune", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
        params, source, _ = tx.resolve_params(None, *_frame())
        assert (params, source) == ({"max_depth": 5}, "cached_stale")

    def test_failed_tuning_without_cache_falls_back_to_library_defaults(self, monkeypatch):
        self._cache(monkeypatch, None)
        monkeypatch.setattr(tx, "tune", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
        params, source, _ = tx.resolve_params(None, *_frame())
        assert (params, source) == (tx.LIBRARY_DEFAULTS, "library_defaults")
