# Monitoring (T10, T12, T13)

Four layers, one dashboard ("Taxi demand API: four layers", http://localhost:3000).

| Layer | Signal | Where |
|---|---|---|
| 1 Infrastructure | P95 latency, predictions/s per model_version, requests by status | `taxi_request_seconds`, `taxi_predictions_total` |
| 2 Input | PSI per feature, last 300 live rows vs the reference period | `taxi_input_psi{feature}` |
| 3 Prediction | PSI of predictions, live mean vs reference mean | `taxi_prediction_psi`, `taxi_prediction_mean` |
| 4 Outcome | WAPE per week (labels arrive late, not live) | `uv run python -m src.drift` |

PSI maths is `src/drift.py` (T11): the gauge and the drift report agree. Act level 0.25.

```bash
cp .env.example .env            # set ADMIN_TOKEN (not change-me)
uv run dvc repro                # builds data/processed/demand.parquet = the PSI reference
docker compose up -d --build    # mlflow, api, prometheus, alertmanager, grafana
curl localhost:8000/metrics     # raw metrics
```

Alerts (`monitoring/prometheus/alerts.yml`, `for: 1m` is a demo value, production ~10m):
ApiDown, SlowRequests, NoModelLoaded, InputDrift, PredictionDrift, PredictionMeanShift.
Test the rules without Docker: `promtool test rules monitoring/prometheus/alerts_test.yml`.

Knobs: `MONITOR_WINDOW` (default 300 rows), `MONITOR_PSI_EVERY` (default 20 rows).
Do not go below ~200 rows: with 10 bins a small window leaves bins empty and inflates the PSI.
