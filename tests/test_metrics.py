"""Unit tests for demand-forecast-demo (synthetic data only)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from main import (
    HOLD_OUT_WEEKS,
    bias_pct,
    evaluate_sku,
    mae,
    main,
    mape,
    moving_average_forecast,
)


def test_mape_perfect():
    y = np.array([10.0, 20.0, 30.0])
    assert mape(y, y) == pytest.approx(0.0)


def test_mape_known():
    y_true = np.array([100.0, 100.0])
    y_pred = np.array([110.0, 90.0])
    assert mape(y_true, y_pred) == pytest.approx(10.0)


def test_mae():
    assert mae(np.array([1.0, 2.0]), np.array([1.0, 4.0])) == pytest.approx(1.0)


def test_bias_pct():
    assert bias_pct(np.array([100.0, 100.0]), np.array([110.0, 110.0])) == pytest.approx(10.0)


def test_moving_average_forecast_shape():
    hist = np.arange(1, 21, dtype=float)
    preds = moving_average_forecast(hist, horizon=HOLD_OUT_WEEKS, window=4)
    assert preds.shape == (HOLD_OUT_WEEKS,)
    assert np.isfinite(preds).all()


def test_evaluate_sku_smoke():
    weeks = pd.date_range("2024-01-01", periods=40, freq="W-MON")
    df = pd.DataFrame(
        {
            "sku_id": "SKU-TEST",
            "sku_name": "Test Widget",
            "week_start": weeks,
            "demand_units": np.linspace(50, 90, len(weeks))
            + np.sin(np.arange(len(weeks)) / 3.0) * 5.0,
        }
    )
    detail, metrics = evaluate_sku(df)
    assert len(detail) == HOLD_OUT_WEEKS
    assert "mape_ma" in metrics
    assert np.isfinite(metrics["mape_ma"])


def test_main_writes_outputs(tmp_path, monkeypatch):
    import main as m

    monkeypatch.setattr(m, "OUT_DIR", tmp_path)
    main()
    assert (tmp_path / "metrics_by_sku.csv").exists()
    assert (tmp_path / "forecast_summary.csv").exists()
