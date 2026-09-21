"""SKU-level demand forecasting demo — moving average + sklearn linear trend.

Synthetic data only. Run: python main.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

DATA_PATH = Path(__file__).resolve().parent / "data" / "weekly_demand.csv"
OUT_DIR = Path(__file__).resolve().parent / "output"
HOLD_OUT_WEEKS = 8
MA_WINDOW = 4


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float))))


def bias_pct(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Positive = over-forecast on average."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.mean(y_true)
    if denom == 0:
        return float("nan")
    return float((np.mean(y_pred) - denom) / denom * 100.0)


def moving_average_forecast(history: np.ndarray, horizon: int, window: int = MA_WINDOW) -> np.ndarray:
    """Recursive trailing MA for `horizon` steps beyond `history`."""
    series = list(history.astype(float))
    preds: list[float] = []
    for _ in range(horizon):
        w = series[-window:] if len(series) >= window else series
        next_val = float(np.mean(w))
        preds.append(next_val)
        series.append(next_val)
    return np.array(preds)


def linear_trend_forecast(history: np.ndarray, horizon: int) -> np.ndarray:
    n = len(history)
    X = np.arange(n).reshape(-1, 1)
    model = LinearRegression()
    model.fit(X, history.astype(float))
    X_future = np.arange(n, n + horizon).reshape(-1, 1)
    return model.predict(X_future)


def evaluate_sku(df_sku: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df_sku = df_sku.sort_values("week_start").reset_index(drop=True)
    if len(df_sku) <= HOLD_OUT_WEEKS + MA_WINDOW:
        raise ValueError(f"Not enough history for {df_sku['sku_id'].iloc[0]}")

    train = df_sku.iloc[:-HOLD_OUT_WEEKS]
    test = df_sku.iloc[-HOLD_OUT_WEEKS:]
    y_train = train["demand_units"].to_numpy()
    y_test = test["demand_units"].to_numpy()

    pred_ma = moving_average_forecast(y_train, HOLD_OUT_WEEKS, MA_WINDOW)
    pred_lr = linear_trend_forecast(y_train, HOLD_OUT_WEEKS)

    detail = test[["week_start", "demand_units"]].copy()
    detail = detail.rename(columns={"demand_units": "actual"})
    detail["sku_id"] = df_sku["sku_id"].iloc[0]
    detail["sku_name"] = df_sku["sku_name"].iloc[0]
    detail["forecast_ma"] = np.round(pred_ma, 1)
    detail["forecast_linear"] = np.round(pred_lr, 1)

    metrics = {
        "sku_id": df_sku["sku_id"].iloc[0],
        "sku_name": df_sku["sku_name"].iloc[0],
        "holdout_weeks": HOLD_OUT_WEEKS,
        "mape_ma": round(mape(y_test, pred_ma), 2),
        "mae_ma": round(mae(y_test, pred_ma), 2),
        "bias_pct_ma": round(bias_pct(y_test, pred_ma), 2),
        "mape_linear": round(mape(y_test, pred_lr), 2),
        "mae_linear": round(mae(y_test, pred_lr), 2),
        "bias_pct_linear": round(bias_pct(y_test, pred_lr), 2),
    }
    return detail, metrics


def main() -> None:
    if not DATA_PATH.exists():
        raise SystemExit(f"Missing data file: {DATA_PATH}. Run: python generate_data.py")

    df = pd.read_csv(DATA_PATH, parse_dates=["week_start"])
    if "data_label" in df.columns and not (df["data_label"] == "DEMO_SAMPLE").all():
        print("Warning: unexpected data_label values present.")

    all_detail: list[pd.DataFrame] = []
    all_metrics: list[dict] = []

    for sku_id, g in df.groupby("sku_id", sort=True):
        detail, metrics = evaluate_sku(g)
        all_detail.append(detail)
        all_metrics.append(metrics)

    metrics_df = pd.DataFrame(all_metrics)
    forecast_df = pd.concat(all_detail, ignore_index=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = OUT_DIR / "metrics_by_sku.csv"
    forecast_path = OUT_DIR / "forecast_summary.csv"
    metrics_df.to_csv(metrics_path, index=False)
    forecast_df.to_csv(forecast_path, index=False)

    print("=" * 72)
    print("DEMAND FORECAST DEMO — synthetic SKU-level baselines")
    print(f"Data: {DATA_PATH.name} | Hold-out: {HOLD_OUT_WEEKS} weeks | MA window: {MA_WINDOW}")
    print("=" * 72)
    print(metrics_df.to_string(index=False))
    print()
    print(
        f"Portfolio MAPE (MA):     {metrics_df['mape_ma'].mean():.2f}%  |  "
        f"Portfolio MAPE (Linear): {metrics_df['mape_linear'].mean():.2f}%"
    )
    print(f"Wrote metrics  → {metrics_path}")
    print(f"Wrote forecasts → {forecast_path}")
    print("Label: DEMO / SAMPLE — not production or employer data.")


if __name__ == "__main__":
    main()
