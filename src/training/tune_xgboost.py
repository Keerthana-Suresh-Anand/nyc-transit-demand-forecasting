"""XGBoost hyperparameter selection: the training pipeline chooses the parameters.

Selection criterion = the task production serves. Each candidate is fit with the
production recipe (``xgb_recipe.fit_recipe``) and scored on weekly iterative 14-day
forecasts over XGB_TUNING_FOLDS validation windows spread across the year — a
one-step-ahead CV score would rank candidates on a task the model never performs,
since its served forecasts feed predicted ridership back into the lag features.

Tuning only ever sees the data it is given (the trainer passes the pre-holdout
rows), so the holdout the promotion gate scores stays out-of-sample.

Lifecycle, mirroring the SARIMAX order cache: the chosen set is persisted in S3 and
reused until it is XGB_RETUNE_DAYS old. A re-tune scores the serving set on the same
folds and replaces it only if the best trial beats it by XGB_RETUNE_MIN_GAIN. With no
serving set (first run), the XGBoost library defaults are the baseline to beat.
"""
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import optuna
import pandas as pd

from src.evaluation.origins import forecast_origins, xgb_origin_forecasts
from src.training.xgb_recipe import fit_recipe
from src.utils.config import (
    FORECAST_HORIZON_DAYS,
    S3_XGB_PARAMS_KEY,
    TEST_DAYS,
    XGB_RETUNE_DAYS,
    XGB_RETUNE_MIN_GAIN,
    XGB_SEARCH_SPACE,
    XGB_TUNING_FOLD_SPACING_DAYS,
    XGB_TUNING_FOLDS,
    XGB_TUNING_TRIALS,
    XGB_TUNING_VAL_DAYS,
)
from src.utils.logger import get_logger
from src.utils.s3_helpers import read_s3_json, write_s3_json

logger = get_logger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# XGBoost's own defaults: the cold-start baseline, and the last-resort fallback when
# neither a persisted set nor a successful search is available.
LIBRARY_DEFAULTS: dict = {}


@dataclass
class TuningResult:
    params: dict
    validation_mae: float
    replaced: bool
    baseline: str  # "serving" or "library_defaults"
    baseline_mae: float
    best_trial_params: dict
    best_trial_mae: float
    fold_end_dates: list[str]
    trials: list[dict] = field(default_factory=list)

    def summary(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k != "trials"}


def fold_ends(n_rows: int) -> list[int]:
    """Exclusive end positions of each fold's data, oldest first. Fold k is fit on
    rows [0 : end_k - XGB_TUNING_VAL_DAYS] and scored on the rest."""
    ends = [n_rows - k * XGB_TUNING_FOLD_SPACING_DAYS for k in range(XGB_TUNING_FOLDS)][::-1]
    if ends[0] - XGB_TUNING_VAL_DAYS <= 2 * TEST_DAYS:
        raise ValueError(f"Not enough rows ({n_rows}) for {XGB_TUNING_FOLDS} tuning folds")
    return ends


def fold_mae(params: dict, df: pd.DataFrame, X: pd.DataFrame, y: pd.Series, end: int) -> float:
    fit_end = end - XGB_TUNING_VAL_DAYS
    model, _ = fit_recipe(params, X.iloc[:fit_end], y.iloc[:fit_end])
    origins = forecast_origins(end, XGB_TUNING_VAL_DAYS, FORECAST_HORIZON_DAYS)
    preds = xgb_origin_forecasts(model, df, [df.index[p] for p in origins], FORECAST_HORIZON_DAYS)
    actual = [y.iloc[p + 1:p + 1 + FORECAST_HORIZON_DAYS].to_numpy() for p in origins]
    return float(np.mean(np.abs(np.concatenate(preds) - np.concatenate(actual))))


def cv_mae(params: dict, df: pd.DataFrame, X: pd.DataFrame, y: pd.Series) -> float:
    return float(np.mean([fold_mae(params, df, X, y, e) for e in fold_ends(len(X))]))


def _suggest(trial: optuna.Trial) -> dict:
    params = {}
    for name, spec in XGB_SEARCH_SPACE.items():
        if spec["type"] == "int":
            params[name] = trial.suggest_int(name, spec["low"], spec["high"])
        else:
            params[name] = trial.suggest_float(name, spec["low"], spec["high"],
                                               log=spec.get("log", False))
    return params


def tune(df: pd.DataFrame, X: pd.DataFrame, y: pd.Series, serving: dict | None,
         n_trials: int = XGB_TUNING_TRIALS) -> TuningResult:
    """Search the space with seeded TPE, then decide against the baseline (the serving
    set, or library defaults on first run) scored on the same folds."""
    ends = fold_ends(len(X))
    baseline_name = "serving" if serving is not None else "library_defaults"
    baseline = serving if serving is not None else LIBRARY_DEFAULTS
    baseline_mae = cv_mae(baseline, df, X, y)
    logger.info(f"Tuning baseline ({baseline_name}): validation MAE {baseline_mae:.4f}M")

    def objective(trial: optuna.Trial) -> float:
        params = _suggest(trial)
        maes = []
        for i, end in enumerate(ends):
            maes.append(fold_mae(params, df, X, y, end))
            trial.report(float(np.mean(maes)), i)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(maes))

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        # Stop a trial early once its running mean over the folds is worse than the
        # median of earlier trials at the same fold.
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1),
    )
    study.optimize(objective, n_trials=n_trials)
    best = study.best_trial
    logger.info(f"Best of {n_trials} trials: validation MAE {best.value:.4f}M {best.params}")

    replaced = best.value < baseline_mae * (1 - XGB_RETUNE_MIN_GAIN)
    if serving is None:
        # First run: nothing is serving, so take whichever scored better.
        replaced = best.value < baseline_mae
    chosen, chosen_mae = (best.params, best.value) if replaced else (baseline, baseline_mae)
    logger.info(
        f"{'Adopting best trial' if replaced else f'Keeping {baseline_name} parameters'} "
        f"(best {best.value:.4f} vs {baseline_name} {baseline_mae:.4f}, "
        f"required gain {XGB_RETUNE_MIN_GAIN:.0%})"
    )
    trials = [{"number": t.number, "state": t.state.name, "value": t.value, "params": t.params}
              for t in study.trials]
    return TuningResult(
        params=dict(chosen), validation_mae=chosen_mae, replaced=replaced,
        baseline=baseline_name, baseline_mae=baseline_mae,
        best_trial_params=best.params, best_trial_mae=best.value,
        fold_end_dates=[str(X.index[e - 1].date()) for e in ends], trials=trials,
    )


def _load_cached(s3) -> tuple[dict, date] | None:
    try:
        data = read_s3_json(s3, S3_XGB_PARAMS_KEY)
        return dict(data["params"]), date.fromisoformat(data["tuned_on"])
    except Exception:
        return None


def _save_cached(s3, result: TuningResult, data_through: str) -> None:
    try:
        write_s3_json(s3, {"params": result.params, "tuned_on": str(date.today()),
                           "data_through": data_through, **result.summary()}, S3_XGB_PARAMS_KEY)
        logger.info(f"Persisted XGBoost parameters to S3: {result.params}")
    except Exception as e:
        logger.warning(f"Could not persist XGBoost parameters to S3: {e}")


def resolve_params(s3, df: pd.DataFrame, X: pd.DataFrame, y: pd.Series
                   ) -> tuple[dict, str, TuningResult | None]:
    """Return (params, source, tuning_result). Reuses the persisted set while it is
    fresh; otherwise tunes and persists. Sources: "cached", "tuned", "cached_stale"
    (tuning failed, expired set kept), "library_defaults" (nothing else available)."""
    cached = _load_cached(s3)
    if cached is not None:
        params, tuned_on = cached
        age = (date.today() - tuned_on).days
        if age < XGB_RETUNE_DAYS:
            logger.info(f"Reusing XGBoost parameters tuned {tuned_on} ({age}d < {XGB_RETUNE_DAYS}d)")
            return params, "cached", None
        logger.info(f"XGBoost parameters are {age}d old (≥ {XGB_RETUNE_DAYS}d) — re-tuning")
    else:
        logger.info("No persisted XGBoost parameters — tuning")

    try:
        result = tune(df, X, y, serving=cached[0] if cached else None)
    except Exception as e:
        if cached is not None:
            logger.warning(f"XGBoost tuning failed ({e}); keeping the expired parameters")
            return cached[0], "cached_stale", None
        logger.warning(f"XGBoost tuning failed ({e}); falling back to library defaults")
        return dict(LIBRARY_DEFAULTS), "library_defaults", None
    _save_cached(s3, result, str(X.index[-1].date()))
    return result.params, "tuned", result


def load_serving_params(s3) -> dict | None:
    """The persisted parameter set, or None. For consumers that must evaluate what
    production serves (the walk-forward backtest)."""
    cached = _load_cached(s3)
    return cached[0] if cached else None
