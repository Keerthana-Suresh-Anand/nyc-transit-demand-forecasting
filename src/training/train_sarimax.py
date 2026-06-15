import pickle
import warnings
from datetime import date

import mlflow
import mlflow.statsmodels
import numpy as np
import pandas as pd
import pmdarima as pm
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.utils.config import (
    GOLD_SARIMA_LOCAL_PATH,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    REPORTS_DIR,
    SARIMAX_MODEL_NAME,
)
from src.utils.logger import get_logger

warnings.filterwarnings("ignore")
logger = get_logger(__name__)

EXOG_COLS = ["temp", "precip", "snow_lag1", "is_holiday"]
TEST_DAYS = 30


def load_data() -> tuple[pd.Series, pd.DataFrame]:
    df = pd.read_parquet(GOLD_SARIMA_LOCAL_PATH)
    df.index = pd.to_datetime(df.index)
    df = df.asfreq("D")
    y = df["daily_ridership"] / 1_000_000
    return y, df[EXOG_COLS]


def scale_exog(
    exog: pd.DataFrame, train_idx: pd.Index, test_idx: pd.Index
) -> tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler]:
    scaler = MinMaxScaler()
    train_exog = pd.DataFrame(
        scaler.fit_transform(exog.loc[train_idx]), index=train_idx, columns=EXOG_COLS
    )
    test_exog = pd.DataFrame(
        scaler.transform(exog.loc[test_idx]), index=test_idx, columns=EXOG_COLS
    )
    return train_exog, test_exog, scaler


def find_best_params(train_y: pd.Series, train_exog: pd.DataFrame) -> tuple:
    logger.info("Starting Auto-ARIMA stepwise search...")
    auto_model = pm.auto_arima(
        train_y, exog=train_exog,
        start_p=0, start_q=0, max_p=3, max_q=3,
        m=7, seasonal=True, start_P=0, D=1,
        error_action="ignore", suppress_warnings=True, stepwise=True,
    )
    logger.info(f"Best model found: SARIMAX{auto_model.order}x{auto_model.seasonal_order}")
    return auto_model.order, auto_model.seasonal_order


def run() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading SARIMA gold data")
    y, exog = load_data()

    split_date = y.index[-TEST_DAYS]
    train_idx = y.loc[: split_date - pd.Timedelta(days=1)].index
    test_idx = y.loc[split_date:].index
    train_y, test_y = y.loc[train_idx], y.loc[test_idx]

    train_exog, test_exog, scaler = scale_exog(exog, train_idx, test_idx)

    best_order, best_seasonal = find_best_params(train_y, train_exog)

    with mlflow.start_run(run_name="sarimax_champion") as run:
        mlflow.set_tag("data_version_date", y.index.max().strftime("%Y-%m-%d"))
        mlflow.set_tag("dataset_row_count", len(y))
        mlflow.set_tag("project_phase", "champion_selection")
        mlflow.set_tag("run_date", str(date.today()))
        mlflow.log_params({
            "order": str(best_order),
            "seasonal_order": str(best_seasonal),
            "exog_cols": str(EXOG_COLS),
            "test_days": TEST_DAYS,
            "search_method": "auto_arima_stepwise",
        })

        logger.info("Training final SARIMAX model on training set")
        final_model = SARIMAX(
            train_y, exog=train_exog,
            order=best_order, seasonal_order=best_seasonal,
            enforce_stationarity=False, enforce_invertibility=False,
        ).fit(disp=False)

        forecast = final_model.get_forecast(steps=len(test_y), exog=test_exog)
        y_pred = forecast.predicted_mean
        conf_int = forecast.conf_int()

        mae = mean_absolute_error(test_y, y_pred)
        rmse = np.sqrt(mean_squared_error(test_y, y_pred))
        mape = mean_absolute_percentage_error(test_y, y_pred)
        mlflow.log_metrics({"mae": mae, "rmse": rmse, "mape": mape, "aic": final_model.aic})
        logger.info(f"Holdout metrics — MAE: {mae:.4f}M  RMSE: {rmse:.4f}M  MAPE: {mape:.2%}")

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(test_y.index, test_y, label="Actual", color="steelblue")
        ax.plot(test_y.index, y_pred, label="SARIMAX forecast", linestyle="--", color="darkorange")
        ax.fill_between(test_y.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1],
                        color="gray", alpha=0.25, label="95% CI")
        ax.set_title(f"SARIMAX Champion | MAE: {mae:.3f}M | MAPE: {mape:.2%}")
        ax.legend()
        plot_path = REPORTS_DIR / "sarimax_champion_forecast.png"
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        mlflow.log_artifact(str(plot_path))

        scaler_path = REPORTS_DIR / "sarimax_scaler.pkl"
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)
        mlflow.log_artifact(str(scaler_path))
        logger.info("Scaler logged as MLflow artifact")

        mlflow.statsmodels.log_model(
            final_model, "sarimax_model",
            registered_model_name=SARIMAX_MODEL_NAME,
        )
        logger.info(f"Model registered: {SARIMAX_MODEL_NAME} (run_id={run.info.run_id})")


if __name__ == "__main__":
    run()
