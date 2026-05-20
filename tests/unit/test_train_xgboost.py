"""Tests for XGBoost training helpers."""
import pandas as pd

from src.training.train_xgboost import TARGET_COL, get_feature_cols


class TestGetFeatureCols:
    def test_excludes_target_column(self):
        df = pd.DataFrame({"daily_ridership": [1], "temp": [2], "precip": [3]})
        result = get_feature_cols(df)
        assert TARGET_COL not in result

    def test_returns_all_non_target_columns(self):
        df = pd.DataFrame({"daily_ridership": [1], "temp": [2], "precip": [3], "snow": [4]})
        result = get_feature_cols(df)
        assert set(result) == {"temp", "precip", "snow"}

    def test_preserves_column_order(self):
        df = pd.DataFrame({"temp": [1], "daily_ridership": [2], "precip": [3], "snow": [4]})
        result = get_feature_cols(df)
        assert result == ["temp", "precip", "snow"]

    def test_returns_empty_list_when_only_target(self):
        df = pd.DataFrame({"daily_ridership": [1, 2, 3]})
        assert get_feature_cols(df) == []

    def test_handles_many_feature_columns(self):
        cols = {f"feat_{i}": [i] for i in range(20)}
        cols["daily_ridership"] = [0]
        df = pd.DataFrame(cols)
        result = get_feature_cols(df)
        assert len(result) == 20
        assert "daily_ridership" not in result
