"""Tests for model evaluation: champion selection and MLflow stage transitions."""
from unittest.mock import MagicMock, patch

import numpy as np

from src.evaluation.evaluate_models import run
from src.utils.config import (
    ENSEMBLE_SARIMAX_WEIGHT,
    ENSEMBLE_XGB_WEIGHT,
    SARIMAX_MODEL_NAME,
    XGBOOST_MODEL_NAME,
)

_PRED = np.array([0.1] * 30)
_ACTUAL = np.array([0.1] * 30)


def _make_version(version: str, stage: str = "None") -> MagicMock:
    v = MagicMock()
    v.version = version
    v.current_stage = stage
    return v


def _mock_mlflow_client(versions=("1",), production_version: str | None = None):
    """Build a mock MLflow client.

    versions: all registered version numbers.
    production_version: which version number is currently in Production stage (if any).
    """
    client = MagicMock()
    client.search_model_versions.return_value = [
        _make_version(v, stage="Production" if v == production_version else "None")
        for v in versions
    ]
    return client


def _s3_patches():
    return [
        patch("src.evaluation.evaluate_models.get_s3_client"),
        patch("src.evaluation.evaluate_models.write_s3_json"),
    ]


def _run_with_metrics(sarimax_mae: float, xgb_mae: float) -> str:
    sarimax_metrics = (sarimax_mae, 0.07, 0.02, 0.01, _PRED, _ACTUAL)
    xgb_metrics = (xgb_mae, 0.08, 0.03, -0.01, _PRED, _ACTUAL)

    with patch("src.evaluation.evaluate_models.evaluate_sarimax", return_value=sarimax_metrics), \
         patch("src.evaluation.evaluate_models.evaluate_xgboost", return_value=xgb_metrics), \
         patch("src.evaluation.evaluate_models.mlflow"), \
         patch("src.evaluation.evaluate_models.MlflowClient") as MockClient, \
         patch("src.evaluation.evaluate_models.get_s3_client"), \
         patch("src.evaluation.evaluate_models.write_s3_json"):
        MockClient.return_value = _mock_mlflow_client()
        return run()


class TestChampionSelection:
    def test_sarimax_wins_when_lower_mae(self):
        winner = _run_with_metrics(sarimax_mae=0.05, xgb_mae=0.08)
        assert winner == SARIMAX_MODEL_NAME

    def test_xgboost_wins_when_lower_mae(self):
        winner = _run_with_metrics(sarimax_mae=0.10, xgb_mae=0.07)
        assert winner == XGBOOST_MODEL_NAME

    def test_sarimax_wins_on_tie(self):
        winner = _run_with_metrics(sarimax_mae=0.06, xgb_mae=0.06)
        assert winner == SARIMAX_MODEL_NAME


class TestModelPromotion:
    def test_promotes_both_models_when_no_prior_production(self):
        """First training run — no Production version exists yet, always promote."""
        metrics = (0.05, 0.07, 0.02, 0.01, _PRED, _ACTUAL)
        with patch("src.evaluation.evaluate_models.evaluate_sarimax", return_value=metrics), \
             patch("src.evaluation.evaluate_models.evaluate_xgboost", return_value=metrics), \
             patch("src.evaluation.evaluate_models.mlflow"), \
             patch("src.evaluation.evaluate_models.MlflowClient") as MockClient, \
             patch("src.evaluation.evaluate_models.get_s3_client"), \
             patch("src.evaluation.evaluate_models.write_s3_json"):
            mock_client = _mock_mlflow_client(versions=("1",), production_version=None)
            MockClient.return_value = mock_client
            run()

        promoted_names = {
            call.kwargs["name"]
            for call in mock_client.transition_model_version_stage.call_args_list
        }
        assert SARIMAX_MODEL_NAME in promoted_names
        assert XGBOOST_MODEL_NAME in promoted_names

    def test_transition_targets_production_stage(self):
        metrics = (0.05, 0.07, 0.02, 0.01, _PRED, _ACTUAL)
        with patch("src.evaluation.evaluate_models.evaluate_sarimax", return_value=metrics), \
             patch("src.evaluation.evaluate_models.evaluate_xgboost", return_value=metrics), \
             patch("src.evaluation.evaluate_models.mlflow"), \
             patch("src.evaluation.evaluate_models.MlflowClient") as MockClient, \
             patch("src.evaluation.evaluate_models.get_s3_client"), \
             patch("src.evaluation.evaluate_models.write_s3_json"):
            mock_client = _mock_mlflow_client(versions=("2",), production_version=None)
            MockClient.return_value = mock_client
            run()

        for call in mock_client.transition_model_version_stage.call_args_list:
            assert call.kwargs["stage"] == "Production"

    def test_skips_transition_when_no_versions_registered(self):
        metrics = (0.05, 0.07, 0.02, 0.01, _PRED, _ACTUAL)
        with patch("src.evaluation.evaluate_models.evaluate_sarimax", return_value=metrics), \
             patch("src.evaluation.evaluate_models.evaluate_xgboost", return_value=metrics), \
             patch("src.evaluation.evaluate_models.mlflow"), \
             patch("src.evaluation.evaluate_models.MlflowClient") as MockClient, \
             patch("src.evaluation.evaluate_models.get_s3_client"), \
             patch("src.evaluation.evaluate_models.write_s3_json"):
            mock_client = _mock_mlflow_client(versions=())
            MockClient.return_value = mock_client
            run()

        mock_client.transition_model_version_stage.assert_not_called()

    def test_promotes_latest_version_when_multiple_exist(self):
        metrics = (0.05, 0.07, 0.02, 0.01, _PRED, _ACTUAL)
        with patch("src.evaluation.evaluate_models.evaluate_sarimax", return_value=metrics), \
             patch("src.evaluation.evaluate_models.evaluate_xgboost", return_value=metrics), \
             patch("src.evaluation.evaluate_models.mlflow"), \
             patch("src.evaluation.evaluate_models.MlflowClient") as MockClient, \
             patch("src.evaluation.evaluate_models.get_s3_client"), \
             patch("src.evaluation.evaluate_models.write_s3_json"):
            mock_client = _mock_mlflow_client(versions=("1", "2", "3"), production_version=None)
            MockClient.return_value = mock_client
            run()

        for call in mock_client.transition_model_version_stage.call_args_list:
            assert call.kwargs["version"] == 3

    def test_promotes_when_new_model_has_lower_mae(self):
        """New version beats Production — should be promoted."""
        new_metrics = (0.04, 0.06, 0.02, 0.01, _PRED, _ACTUAL)
        old_metrics = (0.06, 0.08, 0.03, 0.01, _PRED, _ACTUAL)

        with patch("src.evaluation.evaluate_models.evaluate_sarimax", side_effect=[new_metrics, old_metrics]), \
             patch("src.evaluation.evaluate_models.evaluate_xgboost", side_effect=[new_metrics, old_metrics]), \
             patch("src.evaluation.evaluate_models.mlflow"), \
             patch("src.evaluation.evaluate_models.MlflowClient") as MockClient, \
             patch("src.evaluation.evaluate_models.get_s3_client"), \
             patch("src.evaluation.evaluate_models.write_s3_json"):
            mock_client = _mock_mlflow_client(versions=("1", "2"), production_version="1")
            MockClient.return_value = mock_client
            run()

        promoted_versions = {
            call.kwargs["version"]
            for call in mock_client.transition_model_version_stage.call_args_list
        }
        assert 2 in promoted_versions

    def test_does_not_promote_when_new_model_has_higher_mae(self):
        """New version is worse than Production — should not be promoted."""
        new_metrics = (0.08, 0.10, 0.04, 0.02, _PRED, _ACTUAL)
        old_metrics = (0.05, 0.07, 0.02, 0.01, _PRED, _ACTUAL)

        with patch("src.evaluation.evaluate_models.evaluate_sarimax", side_effect=[new_metrics, old_metrics]), \
             patch("src.evaluation.evaluate_models.evaluate_xgboost", side_effect=[new_metrics, old_metrics]), \
             patch("src.evaluation.evaluate_models.mlflow"), \
             patch("src.evaluation.evaluate_models.MlflowClient") as MockClient, \
             patch("src.evaluation.evaluate_models.get_s3_client"), \
             patch("src.evaluation.evaluate_models.write_s3_json"):
            mock_client = _mock_mlflow_client(versions=("1", "2"), production_version="1")
            MockClient.return_value = mock_client
            run()

        mock_client.transition_model_version_stage.assert_not_called()

    def test_skips_gate_when_latest_already_in_production(self):
        """Latest version is already Production — skip gate."""
        metrics = (0.05, 0.07, 0.02, 0.01, _PRED, _ACTUAL)
        with patch("src.evaluation.evaluate_models.evaluate_sarimax", return_value=metrics), \
             patch("src.evaluation.evaluate_models.evaluate_xgboost", return_value=metrics), \
             patch("src.evaluation.evaluate_models.mlflow"), \
             patch("src.evaluation.evaluate_models.MlflowClient") as MockClient, \
             patch("src.evaluation.evaluate_models.get_s3_client"), \
             patch("src.evaluation.evaluate_models.write_s3_json"):
            mock_client = _mock_mlflow_client(versions=("1", "2"), production_version="2")
            MockClient.return_value = mock_client
            run()

        mock_client.transition_model_version_stage.assert_not_called()


class TestEnsembleBaseline:
    def test_writes_ensemble_mae_to_s3(self):
        """Ensemble MAE should be written to S3 after evaluation."""
        sarimax_pred = np.array([0.10] * 30)
        xgb_pred = np.array([0.20] * 30)
        actual = np.array([0.16] * 30)
        # Use the configured ensemble weights so this test tracks config changes.
        expected_ensemble = ENSEMBLE_SARIMAX_WEIGHT * sarimax_pred + ENSEMBLE_XGB_WEIGHT * xgb_pred
        expected_mae = float(np.mean(np.abs(expected_ensemble - actual)))

        sarimax_metrics = (0.05, 0.07, 0.02, 0.01, sarimax_pred, actual)
        xgb_metrics = (0.08, 0.10, 0.03, -0.01, xgb_pred, actual)

        with patch("src.evaluation.evaluate_models.evaluate_sarimax", return_value=sarimax_metrics), \
             patch("src.evaluation.evaluate_models.evaluate_xgboost", return_value=xgb_metrics), \
             patch("src.evaluation.evaluate_models.mlflow"), \
             patch("src.evaluation.evaluate_models.MlflowClient") as MockClient, \
             patch("src.evaluation.evaluate_models.get_s3_client"), \
             patch("src.evaluation.evaluate_models.write_s3_json") as mock_write:
            MockClient.return_value = _mock_mlflow_client(versions=("1",), production_version=None)
            run()

        written = mock_write.call_args[0][1]
        assert abs(written["ensemble_mae"] - expected_mae) < 1e-6
        assert written["champion_model"] == SARIMAX_MODEL_NAME

    def test_skips_baseline_write_when_no_versions(self):
        """No baseline written when no models are registered."""
        metrics = (0.05, 0.07, 0.02, 0.01, _PRED, _ACTUAL)
        with patch("src.evaluation.evaluate_models.evaluate_sarimax", return_value=metrics), \
             patch("src.evaluation.evaluate_models.evaluate_xgboost", return_value=metrics), \
             patch("src.evaluation.evaluate_models.mlflow"), \
             patch("src.evaluation.evaluate_models.MlflowClient") as MockClient, \
             patch("src.evaluation.evaluate_models.get_s3_client"), \
             patch("src.evaluation.evaluate_models.write_s3_json") as mock_write:
            MockClient.return_value = _mock_mlflow_client(versions=())
            run()

        mock_write.assert_not_called()
