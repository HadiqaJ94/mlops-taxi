# NYC taxi demand: production ML system (MLOps)

Forecast the number of pickups per zone for the next hour. The focus is operations
(tracking, versioning, monitoring, rollback), not model quality.

Data: NYC Taxi & Limousine Commission (TLC) Trip Record Data,
https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page. Raw files are not committed:
`src/data/download.py` plus DVC rebuild them.

## Prerequisites
- Git, Docker Desktop (running), and uv: `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
  (macOS/Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`). uv installs the right Python (3.15) itself.

## First-time setup
```bash
git clone https://github.com/HadiqaJ94/mlops-taxi.git
cd mlops-taxi
uv sync                         # creates .venv from uv.lock (never pip install)
copy .env.example .env          # Windows; macOS/Linux: cp .env.example .env
docker compose up -d mlflow     # MLflow server + UI on http://localhost:5000
uv run dvc repro                # download -> preprocess -> train (logs to MLflow)
uv run pytest
```
Edit `.env` and set your own `ADMIN_TOKEN` (it protects `/admin/reload`).

The DVC remote is the folder `dvc-storage` next to the repo folder (`../dvc-storage`).
`uv run dvc push` saves the data there; `uv run dvc pull` gets it back. That folder only exists
on the PC that pushed: on a fresh clone, `uv run dvc repro` rebuilds the data from the TLC files.

## Daily commands
| Task | Command |
|---|---|
| Add a dependency | `uv add <pkg>` (dev tool: `uv add --dev <pkg>`) |
| Rebuild data and model | `uv run dvc repro` (force training: `uv run dvc repro -f -s train`) |
| Tests and lint | `uv run pytest` and `uv run ruff check .` |
| Model versions and aliases | `uv run python -m src.registry status` |
| Promote / roll back | `uv run python -m src.registry promote --version N` / `uv run python -m src.registry rollback` |
| Run the API | `uv run uvicorn api.app:app --port 8000` (one worker). Docs: http://localhost:8000/docs |
| Serve the new champion | after promote or rollback: `curl -X POST localhost:8000/admin/reload -H "X-Admin-Token: <ADMIN_TOKEN from .env>"` |
| Stop the stack | `docker compose down` |

## API
The API serves `models:/taxi-demand-forecaster@champion`, loaded once at startup.
`MODEL_NAME` and `MODEL_ALIAS` (environment variables) override the model and alias.

| Route | What it does |
|---|---|
| `POST /predict` | `{"zone_id": 237, "target_hour": "2020-02-20T18:00:00", "recent_counts": [...]}` returns the predicted pickups for that hour. `recent_counts`: at least 168 hourly counts for the zone, oldest first, the last one being the hour before `target_hour`. |
| `POST /predict_batch` | `{"items": [<request>, ...]}`, up to 1000 items, one model call |
| `GET /health` | `200 ok` with `model_version`, or `503 degraded` when no model is loaded |
| `POST /admin/reload` | re-reads `@champion` without a restart; needs header `X-Admin-Token` |

Every prediction returns `model_version` and `model_ms` (time inside the model); every response has
`X-Request-ID` and `X-Process-Time-Ms` headers.
Status codes: `422` bad input (the body names the field; also a zone the model does not serve),
`503` no model, `401` wrong or missing admin token.

## Data validation
`dvc repro` runs `src/data/validate.py` between `preprocess` and `train`. If `demand.parquet` breaks a
rule, the stage exits with code 1, the pipeline stops and no model is trained on bad data.
The report is `data/processed/validation_report.json` (every failed rule, with examples).

```bash
uv run python -m src.data.validate
```

- Exact columns and types (an unexpected column, such as `cbd_congestion_fee` from 2025 files, fails).
- `zone_id` is one of the 50 zones in `zones.json`; one row per zone and hour; every zone has every
  hour of Jan-Jun 2020 (the lags use `shift()` on rows, so a gap would make them wrong).
- `target >= 0`; `month` matches `pickup_hour`; `period` is `reference` exactly for Jan-Feb.
- `mean_fare` and `mean_distance` are empty exactly when the hour had no trip.
- At most 1% of fares outside $0-200 and of distances outside 0-100 miles. The TLC files have
  about 0.006% absurd values (a $35,799 mean fare); a broken column (fares x100) fails.

Rules and tolerances are in `params.yaml` under `validate`.

## Git workflow
- Never commit to `master`. Create a branch per task: `git checkout -b feat/<task>`.
- Push, open a pull request, get one review, merge. Commit `uv.lock` and `dvc.lock` when they change.
- Never commit `.env`, `data/`, `mlflow-data/`, `mlruns/` or `.venv/`.

## Design choices
- Time-based split: reference = 2020-01..02 (train/validation), drift = 2020-03..06 (COVID collapse).
- Zero-filled zone x hour grid: an hour with no trip is a 0, not a missing row.
- `src/features.py` is shared by training and the API (no train/serve skew).
- Registry aliases: the API serves `models:/taxi-demand-forecaster@champion`; new versions start as `candidate`.
- Every MLflow run logs params, metrics, git commit, and the SHA-256 of the training data.
- `.gitattributes` forces LF line endings and the pipeline writes its JSON files with LF, so DVC
  hashes (and `dvc.lock`) are identical on Windows, macOS and Linux.
- Metrics: MAE, RMSE, WAPE, compared with a "same hour last week" baseline.
