"""The XGBoost fitting recipe, shared by training, tuning and the walk-forward backtest.

Early-stop on the last TEST_DAYS of the data being fit to choose the tree count, then
refit on all of it with that fixed count, so the most recent month is never lost to
the validation slice. One implementation keeps the tuner, the evaluation model, the
production model and the backtest measuring the same procedure.
"""
import pandas as pd
import xgboost as xgb

from src.utils.config import TEST_DAYS

MAX_TREES = 1000
EARLY_STOPPING_ROUNDS = 50
# Settings that are part of the setup, not hyperparameters to tune.
FIXED_PARAMS = {"eval_metric": "mae", "random_state": 42, "enable_categorical": True}


def early_stop_n(params: dict, X: pd.DataFrame, y: pd.Series) -> int:
    """Tree count chosen by early stopping on the last TEST_DAYS of (X, y)."""
    es = xgb.XGBRegressor(**params, **FIXED_PARAMS, n_estimators=MAX_TREES,
                          early_stopping_rounds=EARLY_STOPPING_ROUNDS)
    es.fit(X.iloc[:-TEST_DAYS], y.iloc[:-TEST_DAYS],
           eval_set=[(X.iloc[-TEST_DAYS:], y.iloc[-TEST_DAYS:])], verbose=False)
    return (es.best_iteration or 0) + 1


def fit_fixed(params: dict, n_estimators: int, X: pd.DataFrame, y: pd.Series) -> xgb.XGBRegressor:
    return xgb.XGBRegressor(**params, **FIXED_PARAMS, n_estimators=n_estimators).fit(
        X, y, verbose=False)


def fit_recipe(params: dict, X: pd.DataFrame, y: pd.Series) -> tuple[xgb.XGBRegressor, int]:
    """Early-stop then refit on all of (X, y). Returns (model, n_estimators)."""
    n = early_stop_n(params, X, y)
    return fit_fixed(params, n, X, y), n
