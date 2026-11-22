# Runbook — roll back the taxi-demand model (T18)

**Goal:** return the live API to the previous known-good model in **under 5 minutes**,
without restarting the service.

On Kubernetes a rollback is `kubectl rollout undo` (swap the container version). This
project runs on Docker Compose, so there is no new container to undo: the thing that
changes is **which model the service serves**. Our rollback moves the MLflow `champion`
alias back to the previous version and tells the running API to reload it.

---

## 1. When to roll back

Roll back when the live model is doing harm and a previous version was known good.
Triggers come from monitoring and drift detection (T10 / T11):

- `src/drift.py` reports **PSI ≥ 0.25** on model inputs (the "act" threshold; 0.10–0.25 is "watch"), or
- weekly **WAPE > 1.5×** the validation WAPE (performance drift), or
- the API error rate or latency breaches a monitoring alert.

**Rollback vs. retrain — decide first.** A broken upstream pipeline looks *exactly* like
drift. Check the data source before retraining; retrain only if the model itself is
affected. **Rollback is the fast, safe first move; retraining is the slower fix.**

## 2. Who may trigger it

Agreed in advance so nobody debates it during an incident: **<project on-call owner — fill in>**.

## 3. Preconditions

- A **previous champion exists** — a version carrying a `replaces` tag. Rollback needs
  somewhere to go (see the one-time rehearsal in §6).
- **`ADMIN_TOKEN`** is set in `.env` (the API's `/admin/reload` requires it).
- The **MLflow server** is up (`docker compose up -d mlflow`).

## 4. Steps (under 5 minutes)

1. **See the current state** — note which version is `champion`:
   ```
   uv run python -m src.registry status
   ```
2. **Roll back the registry** — moves `champion` to the previous version and tags the bad one `rolled_back`:
   ```
   uv run python -m src.registry rollback
   ```
3. **Reload the live API** — serves the restored model, no restart (`curl.exe` is for Windows PowerShell; on macOS/Linux type `curl`):
   ```
   curl.exe -X POST http://localhost:8000/admin/reload -H "X-Admin-Token: <ADMIN_TOKEN from .env>"
   ```
4. **Verify** (this step closes the "forgot to reload" loophole):
   ```
   curl.exe http://localhost:8000/health
   uv run python -m src.registry status
   ```
   - `/health` must report the **restored** `model_version`.
   - `status` must show `champion` on the previous version, and `rolled_back=true` on the bad one.

   If `/health` still shows the bad version, the reload in step 3 did not take — repeat it.

## 5. After a rollback — blameless review

Write a short note (feeds the incident report, T19):

- **Timeline:** when the bad model shipped, when it was noticed, when it was undone.
- **Cause:** what made it possible — not *who* did it.
- **Checks that existed:** which caught it (drift alert? `/health`?), which were missing.
- **Actions:** one process change per finding, each with an owner.

## 6. Rehearsal — do this once, before you need it

> "A rollback you have not tried is not a rollback."

This also creates the promotion history that makes a real rollback possible.

1. Train a second version (registers v2 as `candidate`):
   ```
   uv run dvc repro validate        # creates data/processed/validation_report.json (a T4 dep)
   uv run dvc repro -f -s train     # force a fresh training run -> registers v2
   ```
2. Promote it (moves `champion` to v2 and tags it `replaces=v1`):
   ```
   uv run python -m src.registry promote --version 2
   curl.exe -X POST http://localhost:8000/admin/reload -H "X-Admin-Token: <ADMIN_TOKEN from .env>"
   ```
3. Roll back by following §4. Confirm `champion` returns to v1 and `/health` shows v1.
