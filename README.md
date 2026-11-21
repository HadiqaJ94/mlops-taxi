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
uv run dvc repro                # download -> preprocess -> validate -> train (logs to MLflow)
uv run dvc repro -f -s train    # fresh MLflow: force one training run (see the note below)
uv run python -m src.gate       # REQUIRED: promotes the new @candidate to @champion if it wins
uv run pytest
```
Edit `.env` and set your own `ADMIN_TOKEN` (it protects `/admin/reload`).

> **No champion until the gate has run.** Training only registers a `@candidate`; it never
> promotes. On a fresh clone, if you skip `uv run python -m src.gate`, the registry has no
> `@champion` and the API starts degraded: `/health` and `/predict` answer `503`.
>
> **Why `-f -s train` on a fresh clone:** the rebuilt data is byte-identical to the one in
> `dvc.lock`, so DVC considers `train` up to date and skips it, and your empty MLflow gets no
> model. Forcing the single `train` stage registers one.

The DVC remote is the folder `dvc-storage` next to the repo folder (`../dvc-storage`).
`uv run dvc push` saves the data there; `uv run dvc pull` gets it back. That folder only exists
on the PC that pushed: on a fresh clone, `uv run dvc repro` rebuilds the data from the TLC files.

## Daily commands
| Task | Command |
|---|---|
| Add a dependency | `uv add <pkg>` (dev tool: `uv add --dev <pkg>`) |
| Rebuild data and model | `uv run dvc repro` (force training: `uv run dvc repro -f -s train`), then `uv run python -m src.gate` |
| Decide on a promotion | `uv run python -m src.gate` (decide only: `--dry-run`) |
| Retrain on drift | `uv run python -m src.retrain` (`--force`, `--as-of YYYY-MM-DD`, `--dry-run`) |
| Tests and lint | `uv run pytest` and `uv run ruff check .` |
| Model versions and aliases | `uv run python -m src.registry status` |
| Promote / roll back | `uv run python -m src.registry promote --version N` / `uv run python -m src.registry rollback` |
| Run the API | `uv run uvicorn api.app:app --port 8000` (one worker). Docs: http://localhost:8000/docs |
| Serve the new champion | after promote or rollback: `curl -X POST localhost:8000/admin/reload -H "X-Admin-Token: <ADMIN_TOKEN from .env>"` |
| Build and run the API image | see "Run the API in Docker" below |
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

## Training gate
`src/train.py` trains, logs the run to MLflow and registers the new version as `@candidate`. It
never touches `@champion`. `src/gate.py` decides:

```bash
uv run dvc repro                 # registers a new version as @candidate
uv run python -m src.gate        # promotes it only if it wins; writes reports/gate_report.json
uv run python -m src.gate --dry-run                                  # decide, change nothing
uv run python -m src.gate --val-start 2020-06-24 --val-end 2020-07-01   # another window (used by retraining)
```

The candidate, the current champion and the "same hour last week" baseline are scored on **the same
rows**: a fixed validation window, `params.yaml` -> `gate` (16-29 Feb 2020, the last 14 days of the
reference period, which `train.py` also validates on). The candidate is promoted only if:

1. its WAPE is strictly below the baseline's, and
2. there is no champion yet, or its WAPE is at most `champion WAPE x (1 - min_improvement)`.

`min_improvement: 0.01` (1%): retraining the same data gives the same WAPE to 4 decimals, and a
model that is not measurably better should not replace the one in production (with 0, every
re-run would swap the champion for an identical model).

Every decision is tagged on the candidate version in MLflow (`gate_decision` = `promoted` or
`rejected`, `gate_reason`, `gate_candidate_wape`, `gate_baseline_wape`, `gate_champion_wape`,
`gate_window`). Promotion goes through `registry.promote`, so the `replaces` tag that rollback needs
is set as before. Exit code 0 for both promoted and rejected; 1 on an error (no `@candidate`,
MLflow unreachable, empty window). The gate is not a DVC stage: it depends on the registry state,
which DVC cannot track. After a promotion, reload the running API (`/admin/reload`).

## Run the API in Docker
The image installs only the runtime dependencies from `uv.lock` (`uv sync --locked --no-dev`),
dependencies before code so the layer is cached, and runs one uvicorn worker. No secret is in the
image: `MLFLOW_TRACKING_URI` and `ADMIN_TOKEN` are passed at run time.

```bash
docker compose up -d mlflow                      # the API needs the registry
docker build -t mlops-taxi-api:dev .
docker run --rm -p 8000:8000 -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 -e ADMIN_TOKEN=<token> mlops-taxi-api:dev
curl http://localhost:8000/health                # {"status":"ok","model_version":"1"}
```
Open http://localhost:8000/docs (not `0.0.0.0`). The MLflow server accepts requests from
`localhost`, `host.docker.internal` and `mlflow` only (`--allowed-hosts` in `docker-compose.yml`);
any other host gets `403 Invalid Host header`.

## Drift detection
`src/drift.py` compares a window of data with the reference period the model was trained on
(Jan-Feb 2020) and writes `reports/drift_report.json`. It exits with code 1 when it finds drift,
so a scheduled job or the retraining step can react.

```bash
uv run python -m src.drift                                     # last 7 days of the data
uv run python -m src.drift --start 2020-03-09 --end 2020-03-23  # any window
uv run python -m src.drift --no-model                          # input check only, no MLflow needed
```

| Layer | Check | Drift when |
|---|---|---|
| Input | PSI per feature (`lag_1`, `lag_24`, `lag_168`, `mean_fare`, `mean_distance`) | PSI >= 0.25 |
| Prediction | PSI and mean of the champion's predictions | PSI >= 0.25 |
| Outcome | WAPE per week (weeks start on Monday) | above 1.5x the validation WAPE (0.107) |

Thresholds are in `params.yaml` under `drift` (0.1 = watch, 0.25 = act, course rule of thumb).
The PSI helpers `bin_edges`, `proportions` and `psi_from_profile` are reused by the API's `/metrics` gauge.

What it finds on the TLC data:

| Window | Input PSI (`lag_1`) | Prediction PSI | Weekly WAPE | Drift |
|---|---|---|---|---|
| 2-9 March 2020 | 0.002 | 0.002 | 0.12 | no |
| 9-23 March 2020 (lockdown) | 0.73 | 0.36 | 0.23, then 1.17 | yes |
| 25 May-1 June 2020 | 6.1 | 5.6 | 0.66 | yes |

## Retraining
`src/retrain.py` reacts to drift: it checks the most recent week with `src/drift.py`, and if there
is drift it retrains on the weeks just before, registers the result as `@candidate` and hands it
to the training gate, which compares it with the champion **on that same recent week**.

```bash
uv run python -m src.retrain                    # drift check on the last week of data; retrain only if drift
uv run python -m src.retrain --force            # retrain even without drift (trigger=manual)
uv run python -m src.retrain --as-of 2020-03-23 --force   # as if "now" were 23 March 2020 (incident demo)
uv run python -m src.retrain --dry-run --force  # train and decide in memory: nothing registered or promoted
```

Windows (`params.yaml` -> `retrain`), all ending at `as_of` (excluded; default = end of the data):

| Window | Default | Used for |
|---|---|---|
| validation = `[as_of - val_weeks, as_of)` | 1 week | drift check, candidate metrics, gate |
| training = `[val_start - train_weeks, val_start)` | 8 weeks | fitting the new model |

No leakage: every training row is strictly before the validation week, and lags only look back.
`--train-weeks` / `--val-weeks` override the params; if the training window would start before the
first hour with a full week of lags (8 Jan 2020), the run stops with an error.
`--drift-report FILE` reuses an existing drift report instead of running the check.
Without a champion, the drift check falls back to the input check only (`--no-model`).

Outputs: `reports/retrain_report.json` (drift found or not, features in drift, windows, candidate
version and WAPE, gate decision), plus `reports/drift_report.json` and `reports/gate_report.json`.
The MLflow run is tagged `trigger=drift|manual`, `training_period=recent`, `as_of`, the windows and
the drift report path. Exit code 0 for "no retrain needed", promoted and rejected; 1 on an error.
After a promotion, reload the running API (`/admin/reload`).

What it does on the TLC data:

| `as_of` | Drift | Candidate WAPE | Champion WAPE | Gate |
|---|---|---|---|---|
| 2020-03-09 (`--train-weeks 6`) | no | - | - | no retrain |
| 2020-03-23 (lockdown week) | yes | 0.69 | 1.17 (v1, Jan-Feb) | promoted |
| 2020-07-01 (end of data) | yes | 0.21 | 0.28 (v4, the March retrain) | promoted |

`drift.py` always compares with the original reference period (Jan-Feb 2020), so after a
retraining on recent weeks it keeps reporting input drift; the gate, not the drift flag, is what
protects the champion.

## CI workflows
| Workflow | Trigger | What it runs |
|---|---|---|
| `ci.yml` | every pull request and push to `master` | `ruff check`, `pytest` |
| `train.yml` | manual (input: `min_improvement`) | `dvc repro validate` -> forced `train` stage -> `src.gate` |
| `retrain.yml` | every Monday 06:00 UTC + manual (inputs: `force`, `as_of`) | same setup, reference model gated to champion, then `src.retrain` (drift check -> retrain -> gate) |

Run one: GitHub -> **Actions** -> pick `train` or `retrain` -> **Run workflow** -> fill the inputs.
The reports are attached to the run as **artifacts** (bottom of the run page): `train-reports`
(`metrics.json`, `gate_report.json`, data quality and validation reports) and `retrain-reports`
(`drift_report.json`, `retrain_report.json`, `gate_report.json`). They are uploaded even when a
step fails. A scheduled workflow only runs from the default branch.

How they cope with what a runner does not have:
- **No team MLflow server**: `MLFLOW_TRACKING_URI=sqlite:///mlflow.db`, a registry inside the
  runner, deleted with it. Same code path as local runs (it is `src/registry.py`'s default), and no
  service container, port or `--allowed-hosts` to manage.
- **No DVC remote** (it is a folder on one laptop): `dvc repro` rebuilds the data from the TLC files
  (~300 MB). `actions/cache` keeps `data/raw` between runs, keyed on the download params and
  `src/data/download.py`. Training is forced (`--force --single-item train`), because rebuilt data
  matches `dvc.lock` and DVC would otherwise skip it.
- Drift is not a failure: `src.retrain` runs `src.drift` itself and turns its exit code 1 into a
  decision (retrain or not); the job fails only on a real error.

**Trade-off:** the CI registry starts empty and disappears after the job, so these workflows do not
promote anything the API serves. They prove that the pipeline (data -> train -> gate -> drift ->
retrain -> gate) is reproducible from a clean machine and leave the evidence as artifacts. Real
promotions are made against the team's MLflow server, with the same commands run locally.

## Incident simulation
`src/simulate_incident.py` replays real taxi traffic against the running API: first a normal week,
then the lockdown week. Each request is a real (zone, hour) from `demand.parquet`; its true pickup
count is kept aside as a label that "arrives later", and never sent to the API.

```bash
uv run uvicorn api.app:app --port 8000                       # in another terminal
uv run python -m src.simulate_incident                       # 120 s normal (2-9 March), 180 s drift (16-23 March)
uv run python -m src.simulate_incident --seconds 30 --rate 10                   # quick try
uv run python -m src.simulate_incident --phase drift 2020-04-06 2020-04-13      # custom phase(s)
```

- Requests are shuffled within each phase (seed 42), so any slice of traffic is a fair sample of
  the week; `--in-order` replays hour by hour instead.
- Output: `reports/incident_timeline.csv` (one line per request: time, phase, `request_id`, status,
  `model_version`, predicted, actual) and `reports/incident_summary.json` (per phase). Both are
  generated, not committed.
- Connection errors are logged and the run continues, so it can keep going through a reload,
  a rollback or a restart of the API.

A 20-second run per phase, at 20 requests per second, with model v1:

| Phase | Mean predicted | Mean actual | WAPE |
|---|---|---|---|
| normal (2-9 March) | 176 | 169 | 0.13 |
| drift (16-23 March) | 57 | 27 | 1.15 |

## Git workflow
- Never commit to `master`. Create a branch per task: `git checkout -b feat/<task>`.
- Push, open a pull request, get one review, merge. Commit `uv.lock` and `dvc.lock` when they change.
- Never commit `.env`, `data/`, `mlflow-data/`, `mlruns/` or `.venv/`.

## Design choices
- Time-based split: reference = 2020-01..02 (train/validation), drift = 2020-03..06 (COVID collapse).
- Zero-filled zone x hour grid: an hour with no trip is a 0, not a missing row.
- `src/features.py` is shared by training and the API (no train/serve skew).
- Registry aliases: the API serves `models:/taxi-demand-forecaster@champion`; new versions start as
  `candidate`, and only the training gate (`src/gate.py`) moves `champion` forward.
- Every MLflow run logs params, metrics, git commit, and the SHA-256 of the training data.
- `.gitattributes` forces LF line endings and the pipeline writes its JSON files with LF, so DVC
  hashes (and `dvc.lock`) are identical on Windows, macOS and Linux.
- Metrics: MAE, RMSE, WAPE, compared with a "same hour last week" baseline.
