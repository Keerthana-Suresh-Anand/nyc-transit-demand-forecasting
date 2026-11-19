"""Tests for the champion gate and ensemble weight analysis.

The gate compares the candidate's logged per-origin holdout forecasts with the
Production version re-scored from the origins it can be scored on fairly, plus a
same-origin seasonal-naive floor. Tests patch the
MLflow-facing helpers (and the Production scorer) directly so they exercise the
gating/ensemble logic without a real registry.
"""
import contextlib
import pickle
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.evaluation import evaluate_models
from src.evaluation.evaluate_models import (
    baseline_predictions,
    evaluate_baselines,
    promotion_decision,
    run,
)
from src.utils.config import (
    ENSEMBLE_SARIMAX_WEIGHT,
    ENSEMBLE_XGB_WEIGHT,
    SARIMAX_MODEL_NAME,
    XGBOOST_MODEL_NAME,
)

ORIGINS = pd.to_datetime(["2024-12-31", "2025-01-07", "2025-01-14"])
H = 14


def _holdout(pred: np.ndarray, actual: np.ndarray) -> pd.DataFrame:
    """Per-origin holdout records: three weekly origins x 14 steps (42 rows)."""
    frames = [
        pd.DataFrame({
            "origin": o,
            "date": pd.date_range(o + pd.Timedelta(days=1), periods=H, freq="D"),
            "step": np.arange(1, H + 1),
        })
        for o in ORIGINS
    ]
    return pd.concat(frames, ignore_index=True).assign(y_true=actual, y_pred=pred)


def _gold(level_m: float = 1.0) -> pd.DataFrame:
    """Gold covering the weeks before and through the holdout; a flat level far from
    the holdout's 0.1 makes seasonal-naive a weak floor unless overridden."""
    idx = pd.date_range("2024-11-01", "2025-02-15", freq="D")
    return pd.DataFrame({"daily_ridership": level_m * 1_000_000}, index=idx)


@contextlib.contextmanager
def _patched(*, versions: dict, mae: dict, holdout: dict | None = None,
             prod_after: dict | None = None, prod_mae: dict | None = None,
             gold: pd.DataFrame | None = None, prod_cutoff: dict | None = None):
    """Patch the gate's MLflow-facing helpers.

    versions:   {model_name: (new_ver, prod_ver)}
    mae:        {(model_name, version): mae} — logged holdout MAE per version
    holdout:    {model_name: DataFrame}; defaults to a flat 30-day window per family.
    prod_after: {model_name: version} — what resolve_production_version reports
                after the gates ran; defaults to each family's candidate version
                (i.e. candidates were promoted, baseline and production aligned).
    prod_mae:   {model_name: MAE or None} — the Production version's error when
                re-scored from the holdout origins (None = not scorable). Defaults
                to the Production version's own logged MAE.
    gold:       gold SARIMA frame for the seasonal-naive floor.
    prod_cutoff:{model_name: date} — Production's training cutoff; only origins on
                or after it are scored. Defaults to before every origin.
    """
    prod_cutoff = prod_cutoff or {}
    if holdout is None:
        # Each candidate's rows carry its logged MAE as a constant error.
        actual = np.full(3 * H, 0.1)
        holdout = {
            name: _holdout(actual + (mae.get((name, vers[0])) or 0.0), actual)
            for name, vers in versions.items()
        }
    if prod_after is None:
        prod_after = {name: vers[0] for name, vers in versions.items()}
    if prod_mae is None:
        prod_mae = {name: mae.get((name, vers[1])) for name, vers in versions.items()}

    def score_production(_client, name, _version, origins):
        err = prod_mae.get(name)
        cutoff = pd.Timestamp(prod_cutoff.get(name, "2024-01-01"))
        if err is None:
            return cutoff, {}
        h = holdout[name]
        return cutoff, {o: h.loc[h["origin"] == o, "y_true"].to_numpy() + err
                        for o in origins if o >= cutoff}

    def latest_prod(_client, name):
        return versions.get(name, (None, None))

    def logged_metric(_client, name, version, metric="mae"):
        return mae.get((name, int(version)))

    def load_holdout(_client, name, _version):
        return holdout.get(name)

    def resolve_prod(_client, name):
        ver = prod_after.get(name)
        return SimpleNamespace(version=str(ver)) if ver is not None else None

    with patch("src.evaluation.evaluate_models.mlflow"), \
         patch("src.evaluation.evaluate_models.MlflowClient"), \
         patch("src.evaluation.evaluate_models._latest_and_prod_versions", side_effect=latest_prod), \
         patch("src.evaluation.evaluate_models._logged_metric", side_effect=logged_metric), \
         patch("src.evaluation.evaluate_models._load_holdout", side_effect=load_holdout), \
         patch("src.evaluation.evaluate_models.resolve_production_version", side_effect=resolve_prod), \
         patch("src.evaluation.evaluate_models._promote") as mock_promote, \
         patch("src.evaluation.evaluate_models._score_production", side_effect=score_production), \
         patch("src.evaluation.evaluate_models._load_gold_sarima",
               return_value=_gold() if gold is None else gold), \
         patch("src.evaluation.evaluate_models.get_s3_client"), \
         patch("src.evaluation.evaluate_models.write_s3_json") as mock_write:
        yield mock_promote, mock_write


class TestChampionSelection:
    def test_sarimax_wins_when_lower_mae(self):
        with _patched(
            versions={SARIMAX_MODEL_NAME: (1, None), XGBOOST_MODEL_NAME: (1, None)},
            mae={(SARIMAX_MODEL_NAME, 1): 0.05, (XGBOOST_MODEL_NAME, 1): 0.08},
        ):
            assert run() == SARIMAX_MODEL_NAME

    def test_xgboost_wins_when_lower_mae(self):
        with _patched(
            versions={SARIMAX_MODEL_NAME: (1, None), XGBOOST_MODEL_NAME: (1, None)},
            mae={(SARIMAX_MODEL_NAME, 1): 0.10, (XGBOOST_MODEL_NAME, 1): 0.07},
        ):
            assert run() == XGBOOST_MODEL_NAME

    def test_sarimax_wins_on_tie(self):
        with _patched(
            versions={SARIMAX_MODEL_NAME: (1, None), XGBOOST_MODEL_NAME: (1, None)},
            mae={(SARIMAX_MODEL_NAME, 1): 0.06, (XGBOOST_MODEL_NAME, 1): 0.06},
        ):
            assert run() == SARIMAX_MODEL_NAME


class TestModelPromotion:
    def test_promotes_both_models_when_no_prior_production(self):
        with _patched(
            versions={SARIMAX_MODEL_NAME: (1, None), XGBOOST_MODEL_NAME: (1, None)},
            mae={(SARIMAX_MODEL_NAME, 1): 0.05, (XGBOOST_MODEL_NAME, 1): 0.05},
        ) as (promote, _):
            run()
        promoted_names = {c.args[1] for c in promote.call_args_list}
        assert SARIMAX_MODEL_NAME in promoted_names
        assert XGBOOST_MODEL_NAME in promoted_names

    def test_promotes_latest_version_when_multiple_exist(self):
        with _patched(
            versions={SARIMAX_MODEL_NAME: (3, None), XGBOOST_MODEL_NAME: (3, None)},
            mae={(SARIMAX_MODEL_NAME, 3): 0.05, (XGBOOST_MODEL_NAME, 3): 0.05},
        ) as (promote, _):
            run()
        for c in promote.call_args_list:
            assert c.args[2] == 3

    def test_skips_transition_when_no_versions_registered(self):
        with _patched(
            versions={SARIMAX_MODEL_NAME: (None, None), XGBOOST_MODEL_NAME: (None, None)},
            mae={},
        ) as (promote, write):
            run()
        promote.assert_not_called()
        write.assert_not_called()

    def test_promotes_when_new_model_has_lower_mae(self):
        with _patched(
            versions={SARIMAX_MODEL_NAME: (2, 1), XGBOOST_MODEL_NAME: (2, 1)},
            mae={
                (SARIMAX_MODEL_NAME, 2): 0.04, (SARIMAX_MODEL_NAME, 1): 0.06,
                (XGBOOST_MODEL_NAME, 2): 0.04, (XGBOOST_MODEL_NAME, 1): 0.06,
            },
        ) as (promote, _):
            run()
        promoted_versions = {c.args[2] for c in promote.call_args_list}
        assert promoted_versions == {2}

    def test_does_not_promote_when_new_model_has_higher_mae(self):
        with _patched(
            versions={SARIMAX_MODEL_NAME: (2, 1), XGBOOST_MODEL_NAME: (2, 1)},
            mae={
                (SARIMAX_MODEL_NAME, 2): 0.08, (SARIMAX_MODEL_NAME, 1): 0.05,
                (XGBOOST_MODEL_NAME, 2): 0.08, (XGBOOST_MODEL_NAME, 1): 0.05,
            },
        ) as (promote, _):
            run()
        promote.assert_not_called()

    def test_skips_gate_when_latest_already_in_production(self):
        with _patched(
            versions={SARIMAX_MODEL_NAME: (2, 2), XGBOOST_MODEL_NAME: (2, 2)},
            mae={(SARIMAX_MODEL_NAME, 2): 0.05, (XGBOOST_MODEL_NAME, 2): 0.05},
        ) as (promote, _):
            run()
        promote.assert_not_called()


class TestEnsembleBaseline:
    def test_writes_ensemble_mae_to_s3(self):
        sar = np.full(3 * H, 0.10)
        xgb = np.full(3 * H, 0.20)
        actual = np.full(3 * H, 0.16)
        expected_ensemble = ENSEMBLE_SARIMAX_WEIGHT * sar + ENSEMBLE_XGB_WEIGHT * xgb
        expected_mae = float(np.mean(np.abs(expected_ensemble - actual)))

        with _patched(
            versions={SARIMAX_MODEL_NAME: (1, None), XGBOOST_MODEL_NAME: (1, None)},
            mae={(SARIMAX_MODEL_NAME, 1): 0.05, (XGBOOST_MODEL_NAME, 1): 0.08},
            holdout={
                SARIMAX_MODEL_NAME: _holdout(sar, actual),
                XGBOOST_MODEL_NAME: _holdout(xgb, actual),
            },
        ) as (_, mock_write):
            run()

        written = mock_write.call_args[0][1]
        assert abs(written["ensemble_mae"] - expected_mae) < 1e-6
        assert written["champion_model"] == SARIMAX_MODEL_NAME

    def test_skips_baseline_write_when_no_versions(self):
        with _patched(
            versions={SARIMAX_MODEL_NAME: (None, None), XGBOOST_MODEL_NAME: (None, None)},
            mae={},
        ) as (_, mock_write):
            run()
        mock_write.assert_not_called()


class TestBaselineVersionLineage:
    """The baseline's ensemble MAE always comes from the freshly trained candidates,
    so the JSON must record both the candidate versions and the post-gate Production
    versions — and warn when they differ (a candidate failed its gate)."""

    def test_records_versions_when_aligned(self):
        with _patched(
            versions={SARIMAX_MODEL_NAME: (2, 1), XGBOOST_MODEL_NAME: (2, 1)},
            mae={
                (SARIMAX_MODEL_NAME, 2): 0.04, (SARIMAX_MODEL_NAME, 1): 0.06,
                (XGBOOST_MODEL_NAME, 2): 0.04, (XGBOOST_MODEL_NAME, 1): 0.06,
            },
        ) as (_, mock_write):
            run()
        written = mock_write.call_args[0][1]
        assert written["baseline_versions"] == {"sarimax": 2, "xgboost": 2}
        assert written["production_versions"] == {"sarimax": 2, "xgboost": 2}

    def test_warns_when_candidate_not_promoted(self):
        # XGBoost candidate v2 loses its gate → Production keeps v1, while the
        # baseline ensemble MAE was computed from v2's holdout predictions.
        with _patched(
            versions={SARIMAX_MODEL_NAME: (2, 1), XGBOOST_MODEL_NAME: (2, 1)},
            mae={
                (SARIMAX_MODEL_NAME, 2): 0.04, (SARIMAX_MODEL_NAME, 1): 0.06,
                (XGBOOST_MODEL_NAME, 2): 0.08, (XGBOOST_MODEL_NAME, 1): 0.05,
            },
            prod_after={SARIMAX_MODEL_NAME: 2, XGBOOST_MODEL_NAME: 1},
        ) as (_, mock_write):
            with patch("src.evaluation.evaluate_models.logger") as mock_logger:
                run()

        written = mock_write.call_args[0][1]
        assert written["baseline_versions"] == {"sarimax": 2, "xgboost": 2}
        assert written["production_versions"] == {"sarimax": 2, "xgboost": 1}
        mismatch_warnings = [
            c for c in mock_logger.warning.call_args_list if "not live" in c.args[0]
        ]
        assert len(mismatch_warnings) == 1


class TestHeadToHeadGate:
    """Candidate vs Production on the same holdout window, plus a seasonal-naive floor."""

    def test_regression_older_easy_window_no_longer_blocks_promotion(self):
        # Production's own logged MAE (0.15, from an easier window) is lower than the
        # candidate's (0.22) — the old gate rejected this forever. On the candidate's
        # window Production actually scores 0.30, so the candidate wins.
        with _patched(
            versions={SARIMAX_MODEL_NAME: (23, 16), XGBOOST_MODEL_NAME: (23, 7)},
            mae={(SARIMAX_MODEL_NAME, 23): 0.22, (SARIMAX_MODEL_NAME, 16): 0.15,
                 (XGBOOST_MODEL_NAME, 23): 0.22, (XGBOOST_MODEL_NAME, 7): 0.15},
            prod_mae={SARIMAX_MODEL_NAME: 0.30, XGBOOST_MODEL_NAME: 0.30},
        ) as (promote, write):
            run()
        assert {(c.args[1], c.args[2]) for c in promote.call_args_list} == {
            (SARIMAX_MODEL_NAME, 23), (XGBOOST_MODEL_NAME, 23)}
        gate = write.call_args[0][1]["gate"]["xgboost"]
        assert gate["method"] == "head_to_head" and gate["promoted"] is True
        assert gate["production_mae"] == pytest.approx(0.30)

    def test_unscorable_production_falls_back_to_naive_floor(self):
        with _patched(
            versions={SARIMAX_MODEL_NAME: (2, 1), XGBOOST_MODEL_NAME: (2, 1)},
            mae={(SARIMAX_MODEL_NAME, 2): 0.05, (XGBOOST_MODEL_NAME, 2): 0.05},
            prod_mae={SARIMAX_MODEL_NAME: None, XGBOOST_MODEL_NAME: None},
        ) as (promote, write):
            run()
        assert {c.args[2] for c in promote.call_args_list} == {2}
        assert write.call_args[0][1]["gate"]["sarimax"]["method"] == "naive_floor"

    def test_naive_floor_blocks_candidate_that_loses_to_naive(self):
        # Gold equals the holdout actuals, so seasonal-naive is perfect (MAE 0)
        with _patched(
            versions={SARIMAX_MODEL_NAME: (2, 1), XGBOOST_MODEL_NAME: (2, 1)},
            mae={(SARIMAX_MODEL_NAME, 2): 0.05, (XGBOOST_MODEL_NAME, 2): 0.05},
            prod_mae={SARIMAX_MODEL_NAME: None, XGBOOST_MODEL_NAME: None},
            gold=_gold(level_m=0.1),
        ) as (promote, _):
            run()
        promote.assert_not_called()

    def test_better_candidate_promoted_even_when_both_lose_to_naive(self):
        # Seen on real data (SARIMAX, Sep-Oct 2026): candidate 0.284 < Production 0.293,
        # both above naive 0.264. Naive isn't servable — keeping the worse model is wrong.
        with _patched(
            versions={SARIMAX_MODEL_NAME: (2, 1), XGBOOST_MODEL_NAME: (2, 1)},
            mae={(SARIMAX_MODEL_NAME, 2): 0.05, (XGBOOST_MODEL_NAME, 2): 0.05},
            prod_mae={SARIMAX_MODEL_NAME: 0.09, XGBOOST_MODEL_NAME: 0.09},
            gold=_gold(level_m=0.1),
        ) as (promote, _):
            with patch("src.evaluation.evaluate_models.logger") as mock_logger:
                run()
        assert {c.args[2] for c in promote.call_args_list} == {2}
        assert any("does not beat seasonal-naive" in c.args[0]
                   for c in mock_logger.warning.call_args_list)

    def test_gate_records_origins_and_all_three_maes(self):
        actual = np.full(3 * H, 0.1)
        h = _holdout(actual + 0.04, actual)
        with _patched(
            versions={SARIMAX_MODEL_NAME: (2, 1), XGBOOST_MODEL_NAME: (2, 1)},
            mae={(SARIMAX_MODEL_NAME, 2): 0.04, (SARIMAX_MODEL_NAME, 1): 0.06,
                 (XGBOOST_MODEL_NAME, 2): 0.04, (XGBOOST_MODEL_NAME, 1): 0.06},
            holdout={SARIMAX_MODEL_NAME: h, XGBOOST_MODEL_NAME: h},
        ) as (_, write):
            run()
        g = write.call_args[0][1]["gate"]["sarimax"]
        assert g["origins"] == ["2024-12-31", "2025-01-07", "2025-01-14"]
        assert g["n_points"] == 3 * H
        assert g["candidate_mae"] == pytest.approx(0.04)
        assert g["production_mae"] == pytest.approx(0.06)
        assert g["naive_mae"] == pytest.approx(0.9)

    def test_compares_only_origins_after_production_cutoff(self):
        # Regression: Production trained into part of the holdout (retrains < 30
        # days apart). The old gate gave up and fell back to the naive floor; the
        # origins after its cutoff are still a fair head-to-head.
        actual = np.full(3 * H, 0.1)
        cand_err = np.r_[np.full(H, 0.50), np.full(2 * H, 0.02)]  # bad only on origin 1
        h = _holdout(actual + cand_err, actual)
        with _patched(
            versions={SARIMAX_MODEL_NAME: (2, 1), XGBOOST_MODEL_NAME: (2, 1)},
            mae={(SARIMAX_MODEL_NAME, 2): 0.18, (SARIMAX_MODEL_NAME, 1): 0.05,
                 (XGBOOST_MODEL_NAME, 2): 0.18, (XGBOOST_MODEL_NAME, 1): 0.05},
            holdout={SARIMAX_MODEL_NAME: h, XGBOOST_MODEL_NAME: h},
            prod_mae={SARIMAX_MODEL_NAME: 0.05, XGBOOST_MODEL_NAME: 0.05},
            prod_cutoff={SARIMAX_MODEL_NAME: "2025-01-05", XGBOOST_MODEL_NAME: "2025-01-05"},
        ) as (promote, write):
            run()
        g = write.call_args[0][1]["gate"]["sarimax"]
        assert g["method"] == "head_to_head"
        assert g["origins"] == ["2025-01-07", "2025-01-14"]
        assert g["candidate_mae"] == pytest.approx(0.02)
        assert g["production_cutoff"] == "2025-01-05"
        assert {c.args[2] for c in promote.call_args_list} == {2}

    def test_cutoff_after_every_origin_falls_back_to_naive_floor(self):
        with _patched(
            versions={SARIMAX_MODEL_NAME: (2, 1), XGBOOST_MODEL_NAME: (2, 1)},
            mae={(SARIMAX_MODEL_NAME, 2): 0.05, (SARIMAX_MODEL_NAME, 1): 0.06,
                 (XGBOOST_MODEL_NAME, 2): 0.05, (XGBOOST_MODEL_NAME, 1): 0.06},
            prod_cutoff={SARIMAX_MODEL_NAME: "2025-01-20", XGBOOST_MODEL_NAME: "2025-01-20"},
        ) as (_, write):
            run()
        g = write.call_args[0][1]["gate"]["xgboost"]
        assert g["method"] == "naive_floor"
        assert g["n_points"] == 3 * H


class TestPromotionDecision:
    def test_head_to_head_picks_the_better_servable_model(self):
        assert promotion_decision(0.10, 0.12, 0.20) == (True, "head_to_head")
        assert promotion_decision(0.13, 0.12, 0.20) == (False, "head_to_head")
        assert promotion_decision(0.10, 0.12, 0.09) == (True, "head_to_head")  # naive irrelevant

    def test_naive_floor_when_production_unscored(self):
        assert promotion_decision(0.10, None, 0.20) == (True, "naive_floor")
        assert promotion_decision(0.25, None, 0.20) == (False, "naive_floor")


class TestBaselines:
    def test_naive_paths_use_only_values_known_at_each_origin(self):
        idx = pd.date_range("2024-11-01", "2025-02-15", freq="D")
        y = pd.Series(np.arange(len(idx), dtype=float), index=idx)
        records = _holdout(np.zeros(3 * H), np.zeros(3 * H))
        sn = baseline_predictions(y, records, "seasonal_naive")
        pe = baseline_predictions(y, records, "persistence")
        for i, origin in enumerate(ORIGINS):
            block = slice(i * H, (i + 1) * H)
            assert sn[block].max() <= y[origin]
            assert (pe[block] == y[origin]).all()
            dates = pd.date_range(origin + pd.Timedelta(days=1), periods=H)
            assert list(sn[block][:7]) == [y[d - pd.Timedelta(days=7)] for d in dates[:7]]
            assert list(sn[block][7:]) == [y[d - pd.Timedelta(days=14)] for d in dates[7:]]

    def test_evaluate_baselines_reports_both_benchmarks(self):
        y = pd.Series(1.0, index=pd.date_range("2024-11-01", "2025-02-15", freq="D"))
        records = _holdout(np.zeros(3 * H), np.full(3 * H, 0.8))
        out = evaluate_baselines(y, records)
        assert out["seasonal_naive_m7"]["mae"] == pytest.approx(0.2)
        assert out["persistence"]["n_evaluated"] == 3 * H


def _synthetic_gold(n=120) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=n, freq="D")
    rng = np.random.default_rng(0)
    weekly = np.array([4.0, 4.2, 4.2, 4.2, 3.9, 2.7, 2.1])
    return pd.DataFrame({
        "daily_ridership": (weekly[idx.dayofweek] + rng.normal(0, 0.05, n)) * 1_000_000,
        "temp": 10 + rng.normal(0, 3, n), "precip": np.abs(rng.normal(0, 2, n)),
        "snow": np.zeros(n), "snow_lag1": np.zeros(n), "is_holiday": np.zeros(n, dtype=int),
    }, index=idx)


class TestProductionScorers:
    """Scorers on real (tiny) models: Production must be scored only when its
    training data ends before the window, and through the serving path."""

    def _client(self, run_tags=None):
        client = MagicMock()
        client.get_model_version.return_value = SimpleNamespace(run_id="r1")
        client.get_run.return_value = SimpleNamespace(data=SimpleNamespace(tags=run_tags or {}))
        return client

    def _sarimax(self, gold, train_end, tmp_path):
        from sklearn.preprocessing import MinMaxScaler
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        cols = list(evaluate_models.SARIMAX_EXOG_COLS)
        train = gold.loc[:train_end]
        scaler = MinMaxScaler().fit(train[cols])
        exog = pd.DataFrame(scaler.transform(train[cols]), index=train.index, columns=cols)
        model = SARIMAX(train["daily_ridership"] / 1e6, exog=exog, order=(1, 0, 0),
                        seasonal_order=(1, 0, 0, 7)).fit(disp=False)
        scaler_path = tmp_path / "scaler.pkl"
        scaler_path.write_bytes(pickle.dumps(scaler))
        return model, str(scaler_path)

    def test_sarimax_scored_from_origin_via_reanchor(self, tmp_path):
        gold = _synthetic_gold()
        model, scaler_path = self._sarimax(gold, "2025-03-01", tmp_path)
        origin = pd.Timestamp("2025-03-19")  # 18-day gap to re-anchor
        with patch.object(evaluate_models.mlflow.statsmodels, "load_model", return_value=model), \
             patch.object(evaluate_models.mlflow.artifacts, "download_artifacts", return_value=scaler_path), \
             patch.object(evaluate_models, "_load_gold_sarima", return_value=gold):
            cutoff, preds = evaluate_models._score_sarimax(self._client(), 1, [origin])
        dates = pd.date_range("2025-03-20", periods=14, freq="D")
        actual = gold.loc[dates, "daily_ridership"].to_numpy() / 1e6
        assert cutoff == pd.Timestamp("2025-03-01")
        assert len(preds[origin]) == 14
        assert np.mean(np.abs(preds[origin] - actual)) < 0.5  # tracks the weekly pattern

    def test_sarimax_scores_only_origins_after_its_training_data(self, tmp_path):
        gold = _synthetic_gold()
        model, scaler_path = self._sarimax(gold, "2025-03-10", tmp_path)
        early, late = pd.Timestamp("2025-03-05"), pd.Timestamp("2025-03-19")
        with patch.object(evaluate_models.mlflow.statsmodels, "load_model", return_value=model), \
             patch.object(evaluate_models.mlflow.artifacts, "download_artifacts", return_value=scaler_path), \
             patch.object(evaluate_models, "_load_gold_sarima", return_value=gold):
            _, preds = evaluate_models._score_sarimax(self._client(), 1, [early, late])
            _, none = evaluate_models._score_sarimax(self._client(), 1, [early])
        assert list(preds) == [late]
        assert none == {}

    def test_xgboost_scored_iteratively_only_when_trained_before_window(self, tmp_path):
        import xgboost as xgb

        from src.transformation.preprocess_ml import build_features
        from src.utils.features import cast_categoricals

        df_ml = cast_categoricals(build_features(_synthetic_gold()))
        ml_path = tmp_path / "ml.parquet"
        df_ml.to_parquet(ml_path)
        feats = [c for c in df_ml.columns if c != "daily_ridership"]
        train = df_ml.loc[:"2025-03-01"]
        model = xgb.XGBRegressor(n_estimators=20, max_depth=3, enable_categorical=True,
                                 tree_method="hist").fit(train[feats], train["daily_ridership"] / 1e6)
        origin = pd.Timestamp("2025-03-19")
        with patch.object(evaluate_models.mlflow.xgboost, "load_model", return_value=model), \
             patch.object(evaluate_models, "GOLD_ML_LOCAL_PATH", ml_path):
            _, preds = evaluate_models._score_xgboost(
                self._client({"data_version_date": "2025-03-01"}), 1, [origin])
            assert len(preds[origin]) == 14
            _, late = evaluate_models._score_xgboost(
                self._client({"data_version_date": "2025-03-25"}), 1, [origin])
            assert late == {}
            assert evaluate_models._score_xgboost(self._client({}), 1, [origin]) == (None, {})

    def test_scoring_failure_returns_empty_not_raises(self):
        with patch.dict(evaluate_models._PRODUCTION_SCORERS,
                        {SARIMAX_MODEL_NAME: MagicMock(side_effect=ValueError("feature mismatch"))}):
            out = evaluate_models._score_production(
                MagicMock(), SARIMAX_MODEL_NAME, 1, [pd.Timestamp("2025-01-01")])
        assert out == (None, {})
