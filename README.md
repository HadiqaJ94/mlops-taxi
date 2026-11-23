# NYC taxi demand: production ML system (MLOps)

Forecasts the number of taxi pickups per zone for the next hour, with experiment tracking,
data and model versioning, a FastAPI service, monitoring, drift-triggered retraining and rollback.

Data: NYC Taxi & Limousine Commission (TLC) Trip Record Data,
https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page. Raw files are not committed:
`src/data/download.py` plus DVC rebuild them.

Architecture: [`docs/architecture.png`](docs/architecture.png). Rollback runbook:
[`docs/runbook.md`](docs/runbook.md). Incident report: [`reports/incident_report.md`](reports/incident_report.md).

## 1. Prerequisites
- Git
- Docker Desktop, running
- uv. Windows PowerShell: `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`.
  macOS/Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`. uv installs Python 3.15 itself.

## 2. First-time setup
```bash
git clone https://github.com/HadiqaJ94/mlops-taxi.git
cd mlops-taxi
uv sync                          # creates .venv from uv.lock (never pip install)
copy .env.example .env           # Windows; macOS/Linux: cp .env.example .env
uv run python -c "import secrets; print(secrets.token_hex(16))"   # paste it as ADMIN_TOKEN in .env
docker compose up -d mlflow      # MLflow server and UI: http://localhost:5000
uv run dvc repro                 # download -> preprocess -> validate -> train (~300 MB the first time)
uv run dvc repro -f -s train     # on a fresh MLflow: force one training run
uv run python -m src.gate        # REQUIRED: promotes @candidate to @champion if it wins
uv run pytest
```
- Without `src.gate` there is no `@champion`, and the API answers `503`.
- The API refuses the default `ADMIN_TOKEN` (`change-me`): set your own in `.env`.

**Windows:**
- Clone into a short path (e.g. `C:\dev\mlops-taxi`). Long paths make `dvc repro` fail with `[WinError 3]`.
- In PowerShell 5.1, type `curl.exe`, not `curl`.
- `SSLError: CERTIFICATE_VERIFY_FAILED` during the download: an antivirus is inspecting HTTPS.
  Pause that scan, or set `REQUESTS_CA_BUNDLE` to its root certificate.

## 3. Run the full stack
```bash
docker compose up -d --build     # mlflow, api, prometheus, alertmanager, grafana
```

| Service | URL |
|---|---|
| API (interactive docs) | http://localhost:8000/docs |
| MLflow | http://localhost:5000 |
| Prometheus | http://localhost:9090 |
| Alertmanager | http://localhost:9093 |
| Grafana dashboard | http://localhost:3000 (anonymous view; admin password `GRAFANA_ADMIN_PASSWORD` in `.env`, default `admin`) |

Stop everything: `docker compose down`.

## 4. Run the API alone
```bash
# locally (MLflow must be running)
uv run uvicorn api.app:app --port 8000

# or in Docker
docker build -t mlops-taxi-api:dev .
docker run --rm -p 8000:8000 -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 -e ADMIN_TOKEN=<token> mlops-taxi-api:dev
```
Open http://localhost:8000/docs (not `0.0.0.0`).

| Route | Use |
|---|---|
| `POST /predict` | `{"zone_id": 237, "target_hour": "2020-02-20T18:00:00", "recent_counts": [<at least 168 hourly counts, oldest first>]}` |
| `POST /predict_batch` | `{"items": [<request>, ...]}`, up to 1000 items |
| `GET /health` | `200` with `model_version`, or `503` when no model is loaded |
| `GET /metrics` | Prometheus metrics |
| `POST /admin/reload` | loads the current `@champion` without a restart; header `X-Admin-Token` |

```bash
curl.exe http://localhost:8000/health
curl.exe -X POST http://localhost:8000/admin/reload -H "X-Admin-Token: <ADMIN_TOKEN>"
```

## 5. Model operations
| Task | Command |
|---|---|
| Rebuild data and retrain | `uv run dvc repro` (force training: `uv run dvc repro -f -s train`) |
| Promote if the candidate wins | `uv run python -m src.gate` (decide only: `--dry-run`) |
| Versions and aliases | `uv run python -m src.registry status` |
| Promote a given version | `uv run python -m src.registry promote --version N` |
| Roll back to the previous champion | `uv run python -m src.registry rollback` |

After a promotion or a rollback, call `/admin/reload` so the running API serves the new champion.

## 6. Drift detection and retraining
```bash
uv run python -m src.drift                                      # last 7 days of the data
uv run python -m src.drift --start 2020-03-09 --end 2020-03-23  # any window
uv run python -m src.drift --no-model                           # input check only, no MLflow needed

uv run python -m src.retrain                                    # retrain only if drift is found
uv run python -m src.retrain --as-of 2020-03-23 --force         # as if "now" were 23 March 2020
uv run python -m src.retrain --dry-run --force                  # nothing registered or promoted
```
Reports are written to `reports/` (`drift_report.json`, `retrain_report.json`, `gate_report.json`).

## 7. Incident simulation
The API must be running (section 3 or 4).
```bash
uv run python -m src.simulate_incident                          # 120 s normal week, then 180 s lockdown week
uv run python -m src.simulate_incident --seconds 30 --rate 10   # quick try
```
Output: `reports/incident_timeline.csv` and `reports/incident_summary.json`. Stop it with Ctrl + C.

## 8. Load test
```bash
docker compose up -d --build
uv run locust -f locustfile.py --host http://localhost:8000     # UI: http://localhost:8089
```

## 9. Tests and lint
```bash
uv run pytest
uv run ruff check .
promtool test rules monitoring/prometheus/alerts_test.yml       # alert rules (needs promtool)
```

## 10. CI/CD (GitHub Actions)
| Workflow | When it runs | How to start it by hand |
|---|---|---|
| `ci.yml` (lint, tests) | every pull request and push to `master` | - |
| `docker.yml` (build, smoke test, push to GHCR) | every pull request and push to `master` | Actions -> docker -> Run workflow |
| `train.yml` (train, gate) | manual | Actions -> train -> Run workflow |
| `retrain.yml` (drift check, retrain, gate) | every Monday 06:00 UTC | Actions -> retrain -> Run workflow |

Reports are attached to each run as artifacts.
