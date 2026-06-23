"""Tests for the champion gate and ensemble weight analysis.

The gate reads each version's logged holdout MAE and per-day holdout predictions
from MLflow rather than re-forecasting (production models are refit on full data,
so re-forecasting them would be in-sample). Tests patch the MLflow-facing helpers
directly so they exercise the gating/ensemble logic without a real registry.
"""
import contextlib
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.evaluation.evaluate_models import run
from src.utils.config import (
    ENSEMBLE_SARIMAX_WEIGHT,
    ENSEMBLE_XGB_WEIGHT,
    SARIMAX_MODEL_NAME,
    XGBOOST_MODEL_NAME,
)


def _holdout(pred: np.ndarray, actual: np.ndarray) -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=len(pred), freq="D").strftime("%Y-%m-%d")
    return pd.DataFrame({"date": dates, "y_true": actual, "y_pred": pred})


@contextlib.contextmanager
def _patched(*, versions: dict, mae: dict, holdout: dict | None = None):
    """Patch the gate's MLflow-facing helpers.

    versions: {model_name: (new_ver, prod_ver)}
    mae:      {(model_name, version): mae}
    holdout:  {model_name: DataFrame}; defaults to a flat 30-day window per family.
    """
    if holdout is None:
        flat = _holdout(np.full(30, 0.1), np.full(30, 0.1))
        holdout = {SARIMAX_MODEL_NAME: flat, XGBOOST_MODEL_NAME: flat}

    def latest_prod(_client, name):
        return versions.get(name, (None, None))

    def logged_metric(_client, name, version, metric="mae"):
        return mae.get((name, int(version)))

    def load_holdout(_client, name, _version):
        return holdout.get(name)

    with patch("src.evaluation.evaluate_models.mlflow"), \
         patch("src.evaluation.evaluate_models.MlflowClient"), \
         patch("src.evaluation.evaluate_models._latest_and_prod_versions", side_effect=latest_prod), \
         patch("src.evaluation.evaluate_models._logged_metric", side_effect=logged_metric), \
         patch("src.evaluation.evaluate_models._load_holdout", side_effect=load_holdout), \
         patch("src.evaluation.evaluate_models._promote") as mock_promote, \
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
        sar = np.full(30, 0.10)
        xgb = np.full(30, 0.20)
        actual = np.full(30, 0.16)
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
