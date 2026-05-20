"""Tests for model evaluation: champion selection and MLflow stage transitions."""
from unittest.mock import MagicMock, patch

from src.evaluation.evaluate_models import run
from src.utils.config import SARIMAX_MODEL_NAME, XGBOOST_MODEL_NAME


def _mock_mlflow_client(versions=("1",)):
    client = MagicMock()
    client.search_model_versions.return_value = [MagicMock(version=v) for v in versions]
    return client


def _run_with_metrics(sarimax_mae: float, xgb_mae: float) -> str:
    sarimax_metrics = (sarimax_mae, 0.07, 0.02, 0.01)
    xgb_metrics = (xgb_mae, 0.08, 0.03, -0.01)

    with patch("src.evaluation.evaluate_models.evaluate_sarimax", return_value=sarimax_metrics), \
         patch("src.evaluation.evaluate_models.evaluate_xgboost", return_value=xgb_metrics), \
         patch("src.evaluation.evaluate_models.mlflow"), \
         patch("src.evaluation.evaluate_models.MlflowClient") as MockClient:
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
    def test_both_models_transitioned_to_production(self):
        with patch("src.evaluation.evaluate_models.evaluate_sarimax", return_value=(0.05, 0.07, 0.02, 0.01)), \
             patch("src.evaluation.evaluate_models.evaluate_xgboost", return_value=(0.08, 0.10, 0.03, -0.01)), \
             patch("src.evaluation.evaluate_models.mlflow"), \
             patch("src.evaluation.evaluate_models.MlflowClient") as MockClient:
            mock_client = _mock_mlflow_client(versions=("1",))
            MockClient.return_value = mock_client
            run()

        transition_calls = mock_client.transition_model_version_stage.call_args_list
        promoted_names = {call.kwargs["name"] for call in transition_calls}
        assert SARIMAX_MODEL_NAME in promoted_names
        assert XGBOOST_MODEL_NAME in promoted_names

    def test_transition_targets_production_stage(self):
        with patch("src.evaluation.evaluate_models.evaluate_sarimax", return_value=(0.05, 0.07, 0.02, 0.01)), \
             patch("src.evaluation.evaluate_models.evaluate_xgboost", return_value=(0.08, 0.10, 0.03, -0.01)), \
             patch("src.evaluation.evaluate_models.mlflow"), \
             patch("src.evaluation.evaluate_models.MlflowClient") as MockClient:
            mock_client = _mock_mlflow_client(versions=("2",))
            MockClient.return_value = mock_client
            run()

        for call in mock_client.transition_model_version_stage.call_args_list:
            assert call.kwargs["stage"] == "Production"

    def test_skips_transition_when_no_versions_registered(self):
        with patch("src.evaluation.evaluate_models.evaluate_sarimax", return_value=(0.05, 0.07, 0.02, 0.01)), \
             patch("src.evaluation.evaluate_models.evaluate_xgboost", return_value=(0.08, 0.10, 0.03, -0.01)), \
             patch("src.evaluation.evaluate_models.mlflow"), \
             patch("src.evaluation.evaluate_models.MlflowClient") as MockClient:
            mock_client = _mock_mlflow_client(versions=())
            MockClient.return_value = mock_client
            run()

        mock_client.transition_model_version_stage.assert_not_called()

    def test_promotes_latest_version_when_multiple_exist(self):
        with patch("src.evaluation.evaluate_models.evaluate_sarimax", return_value=(0.05, 0.07, 0.02, 0.01)), \
             patch("src.evaluation.evaluate_models.evaluate_xgboost", return_value=(0.08, 0.10, 0.03, -0.01)), \
             patch("src.evaluation.evaluate_models.mlflow"), \
             patch("src.evaluation.evaluate_models.MlflowClient") as MockClient:
            mock_client = _mock_mlflow_client(versions=("1", "2", "3"))
            MockClient.return_value = mock_client
            run()

        for call in mock_client.transition_model_version_stage.call_args_list:
            assert call.kwargs["version"] == 3
