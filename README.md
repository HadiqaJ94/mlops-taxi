# NYC taxi demand: production ML system (MLOps)

Forecast pickups per zone for the next hour. The focus is operations (tracking, versioning,
monitoring, rollback), not model quality.

Data: NYC Taxi & Limousine Commission (TLC) Trip Record Data,
https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page (raw files are not committed).

## Quickstart

```bash
uv sync                                  # create .venv and install locked deps
cp .env.example .env
uv run dvc repro                         # download -> preprocess -> train (logs to MLflow)
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db   # look at runs and the registry
uv run python -m src.registry status     # model versions and aliases
uv run python -m src.registry rollback   # champion goes back to the previous version
uv run pytest
```

## Design choices
- Time-based split: reference = 2020-01..02 (train/val), drift = 2020-03..06.
- Zero-filled zone x hour grid (a missing row would be a missing zero).
- `src/features.py` is shared by training and the API (no train/serve skew).
- Registry aliases: the API serves `models:/taxi-demand-forecaster@champion`.
- Every MLflow run logs params, metrics, git commit, and data SHA-256.

## TODO
API (FastAPI) + integration tests, drift monitoring, retraining trigger, incident report.
