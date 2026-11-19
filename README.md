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
The DVC remote is the folder `dvc-storage` next to the repo folder (`../dvc-storage`).
`uv run dvc push` saves the data there; `uv run dvc pull` gets it back.

## Daily commands
| Task | Command |
|---|---|
| Add a dependency | `uv add <pkg>` (dev tool: `uv add --dev <pkg>`) |
| Rebuild data and model | `uv run dvc repro` (force training: `uv run dvc repro -f -s train`) |
| Tests and lint | `uv run pytest` and `uv run ruff check .` |
| Model versions and aliases | `uv run python -m src.registry status` |
| Promote / roll back | `uv run python -m src.registry promote --version N` / `uv run python -m src.registry rollback` |
| Stop the stack | `docker compose down` |

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
- Metrics: MAE, RMSE, WAPE, compared with a "same hour last week" baseline.
