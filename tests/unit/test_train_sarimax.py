"""Tests for SARIMAX training helpers."""
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.training.train_sarimax import EXOG_COLS, scale_exog


def _make_exog(periods: int = 100) -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=periods, freq="D")
    return pd.DataFrame(
        {
            "temp": [20.0 + i * 0.5 for i in range(periods)],
            "precip": [0.0 if i % 7 != 0 else 1.5 for i in range(periods)],
            "snow_lag1": [0.0] * periods,
            "is_holiday": [0] * periods,
        },
        index=dates,
    )


class TestScaleExog:
    def test_returns_three_values(self):
        exog = _make_exog(100)
        train_idx = exog.index[:70]
        test_idx = exog.index[70:]
        result = scale_exog(exog, train_idx, test_idx)
        assert len(result) == 3

    def test_train_values_in_unit_range(self):
        exog = _make_exog(100)
        train_idx = exog.index[:70]
        test_idx = exog.index[70:]
        train_exog, _, _ = scale_exog(exog, train_idx, test_idx)
        assert train_exog.min().min() >= -1e-9
        assert train_exog.max().max() <= 1.0 + 1e-9

    def test_column_names_preserved(self):
        exog = _make_exog(100)
        train_idx = exog.index[:70]
        test_idx = exog.index[70:]
        train_exog, test_exog, _ = scale_exog(exog, train_idx, test_idx)
        assert list(train_exog.columns) == EXOG_COLS
        assert list(test_exog.columns) == EXOG_COLS

    def test_train_index_preserved(self):
        exog = _make_exog(100)
        train_idx = exog.index[:70]
        test_idx = exog.index[70:]
        train_exog, _, _ = scale_exog(exog, train_idx, test_idx)
        assert list(train_exog.index) == list(train_idx)

    def test_test_index_preserved(self):
        exog = _make_exog(100)
        train_idx = exog.index[:70]
        test_idx = exog.index[70:]
        _, test_exog, _ = scale_exog(exog, train_idx, test_idx)
        assert list(test_exog.index) == list(test_idx)

    def test_returns_fitted_scaler(self):
        exog = _make_exog(100)
        train_idx = exog.index[:70]
        test_idx = exog.index[70:]
        _, _, scaler = scale_exog(exog, train_idx, test_idx)
        assert isinstance(scaler, MinMaxScaler)

    def test_train_and_test_shape_matches_index_lengths(self):
        exog = _make_exog(100)
        train_idx = exog.index[:70]
        test_idx = exog.index[70:]
        train_exog, test_exog, _ = scale_exog(exog, train_idx, test_idx)
        assert len(train_exog) == 70
        assert len(test_exog) == 30
