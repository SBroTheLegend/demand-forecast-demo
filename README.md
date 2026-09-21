# Demand Forecast Demo

**SKU-level demand forecasting baseline for supply-chain planning** — synthetic portfolio sample.

Senior Program Manager context: this repo shows how I frame a forecasting workstream — clear baselines, measurable accuracy (MAPE / bias), and outputs planners can challenge. It is **not** production IP; all data is synthetic.

---

## What this demonstrates

- Weekly demand for a small set of fake product SKUs
- Two baselines: trailing moving average and optional scikit-learn linear trend model
- Hold-out evaluation with **MAPE**, **MAE**, and **forecast bias**
- A single entry point: `python main.py`

Production forecasting stacks (hierarchical models, causal features, ERP integration) stay inside employers. This sample is intentionally simple and auditable.

---

## Quick start

```bash
cd demand-forecast-demo
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

Expected output: per-SKU metrics table printed to the console, plus `output/forecast_summary.csv` and `output/metrics_by_sku.csv`.

---

## Data

| File | Description |
|---|---|
| `data/weekly_demand.csv` | Synthetic weekly units by SKU (label: **DEMO / SAMPLE**) |

Generated once via `python generate_data.py` (re-runnable; fixed seed for reproducibility).

---

## Method (high level)

1. Load weekly demand; sort by SKU and week.
2. For each SKU, hold out the last `N` weeks (`N=8` by default).
3. **Moving average** baseline: mean of the prior `window` weeks (default 4).
4. **Linear trend** (optional sklearn): week index → demand on the training window; predict hold-out.
5. Report MAPE, MAE, and % bias; write forecasts and metrics to `output/`.

---

## Project layout

```text
demand-forecast-demo/
├── README.md
├── requirements.txt
├── generate_data.py
├── main.py
├── data/
│   └── weekly_demand.csv
└── output/          # created on run
```

---

## Disclaimer

Synthetic demonstration only. No employer, customer, or proprietary Honeywell (or other) data. Suitable for recruiter review of approach and communication — not a claim of production model performance.
