"""Generate synthetic weekly SKU demand for the portfolio demo.

All values are fake. Label: DEMO / SAMPLE. Fixed seed for reproducibility.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_WEEKS = 104  # ~2 years
OUT_PATH = Path(__file__).resolve().parent / "data" / "weekly_demand.csv"

# Fake catalog — not real products
SKUS = [
    {"sku_id": "SKU-1001", "sku_name": "Alpha Widget A", "base": 420, "trend": 0.35, "season_amp": 55, "noise": 28},
    {"sku_id": "SKU-1002", "sku_name": "Beta Component B", "base": 180, "trend": -0.12, "season_amp": 30, "noise": 18},
    {"sku_id": "SKU-1003", "sku_name": "Gamma Assembly C", "base": 95, "trend": 0.08, "season_amp": 22, "noise": 12},
    {"sku_id": "SKU-1004", "sku_name": "Delta Spare D", "base": 260, "trend": 0.20, "season_amp": 40, "noise": 22},
    {"sku_id": "SKU-1005", "sku_name": "Epsilon Kit E", "base": 55, "trend": 0.05, "season_amp": 15, "noise": 8},
]


def main() -> None:
    rng = np.random.default_rng(SEED)
    start = pd.Timestamp("2023-01-02")  # Monday
    weeks = pd.date_range(start, periods=N_WEEKS, freq="W-MON")

    rows: list[dict] = []
    for sku in SKUS:
        t = np.arange(N_WEEKS, dtype=float)
        seasonal = sku["season_amp"] * np.sin(2 * np.pi * t / 52.0)
        noise = rng.normal(0.0, sku["noise"], size=N_WEEKS)
        demand = sku["base"] + sku["trend"] * t + seasonal + noise
        demand = np.clip(np.round(demand), 1, None).astype(int)
        for week, units in zip(weeks, demand):
            rows.append(
                {
                    "sku_id": sku["sku_id"],
                    "sku_name": sku["sku_name"],
                    "week_start": week.strftime("%Y-%m-%d"),
                    "demand_units": int(units),
                    "data_label": "DEMO_SAMPLE",
                }
            )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(df)} rows → {OUT_PATH}")


if __name__ == "__main__":
    main()
