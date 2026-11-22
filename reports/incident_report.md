# Incident report (T19): demand collapse, March 2020

Run on 22 November 2026, on master after PR #25 (monitoring follow-up), full stack from
`docker compose up` (MLflow, API, Prometheus, Alertmanager, Grafana), traffic from
`src/simulate_incident.py` at 20 requests/s. Champion at the start: **v1**, trained on
January–February 2020.

## Summary

| | Before (v1, normal week) | Incident (v1, lockdown week) | After (v6, lockdown week) |
|---|---|---|---|
| Replayed window | 2–9 March 2020 | 16–23 March 2020 | 16–23 March 2020 |
| Requests / errors | 2 328 / 0 | 3 555 / 0 | 3 510 / 0 |
| Mean predicted / actual pickups | 178.2 / 173.6 | 56.1 / 25.9 | 43.3 / 26.1 |
| **WAPE** | **0.12** | **1.17** | **0.68** |

The API never failed (0 errors, no 5xx). The model failed silently: in the lockdown week it
predicted more than twice the real demand. Only the drift monitoring could see it live, because
the true counts (labels) arrive later.

## Timeline (local time)

| Time | Event | Source |
|---|---|---|
| 20:36:47 | API starts with v1. Prediction gauges are NaN until 300 rows arrive: no alert | API log, Prometheus |
| 20:37:10 | Normal phase starts (2–9 March replayed): PSI stays low, no alert | simulator |
| 20:39:10 | **Drift phase starts** (16–23 March: lockdown) | simulator |
| 20:39:26 | `InputDrift` lag_1 and lag_24 **pending** (PSI above 0.25) | Prometheus |
| 20:39:31 | `PredictionDrift`, `PredictionMeanShift` pending | Prometheus |
| 20:40:26 | **`InputDrift` lag_1, lag_24 firing**, sent to Alertmanager | Alertmanager |
| 20:40:31 | **`PredictionDrift`, `PredictionMeanShift` firing** | Alertmanager |
| 20:42:10 | Drift phase ends (WAPE 1.17 once labels are joined). Response: `uv run python -m src.retrain --as-of 2020-03-23` | operator |
| 20:42:56 | Drift confirmed, candidate v6 trained on 20 Jan–16 Mar, gate **promoted v6** (WAPE 0.69 vs champion 1.17). `POST /admin/reload`: API serves v6, no restart, no downtime | retrain report, API |
| 20:42:56 | Monitor switches its reference to v6's training window (67 200 rows) and resets the gauges | API log |
| 20:43:10 | **All alerts resolved** | Prometheus |
| 20:43–20:46 | Recovery phase: lockdown week replayed on v6, WAPE 0.68 | simulator |
| 20:44:21 | `InputDrift`, `PredictionDrift` firing again (see "What did not go well") | Prometheus |

**Time to detect:** 76 s from the start of the drift to the first firing alert
(~15 s to fill the PSI window, then the `for: 1m` rule delay).
**Time to mitigate:** 2 min 30 s from the alert to v6 serving (the retrain itself took 46 s).

## Detection: what each layer saw

| Layer | Signal | Normal | Drift |
|---|---|---|---|
| 1 Infrastructure | errors, latency | 0 errors | 0 errors: infrastructure looks healthy |
| 2 Input | PSI lag_1 / lag_24 / lag_168 | < 0.1 | ~3.3 / ~2.6 / < 0.25 (lag_168 is still the pre-lockdown week) |
| 3 Prediction | PSI, mean vs reference 171.6 | normal | PSI ~1.9, mean ~56 |
| 4 Outcome | WAPE (labels, later) | 0.12 | 1.17 |

Layers 2 and 3 caught the incident live; layer 1 alone would have missed it.

## Root cause

Covid-19 lockdown in New York (from 16 March 2020): taxi demand collapsed by ~85%. v1 learned
January–February patterns; its lag_24 and lag_168 features still pointed to normal demand, so it
over-predicted. This is concept and data drift, not a bug.

## Response

1. Drift alerts confirmed by the offline drift report (`src/drift.py`): lag_1, lag_24 and mean_fare
   PSI > 0.25, WAPE 1.17 > 0.16.
2. Retraining on the most recent 8 weeks (T16), validated on the lockdown week.
3. Gate (T6) promoted v6 only because it beat both the baseline and the champion by more than 1%.
4. Hot reload of the champion alias. The PSI reference follows the new model (MLflow tag
   `train_window`). Rollback stays available (`docs/runbook.md`): v1 is kept in the registry.

## What went well

- Detection in 76 s, before any label was available, from two independent layers.
- From alert to a better model in 2 min 30 s, with the gate as a safety net and no downtime.
- The first rehearsal (earlier the same day) found two monitoring issues, fixed in PR #25 and
  checked in this run: no false `PredictionMeanShift` at startup (gauges are NaN until data
  arrives), and the PSI reference now follows the champion, so alerts resolve on reload.

## What did not go well (lessons and follow-ups)

1. **Alerts fired again 70 s after the fix.** This is correct, not a bug: v6 was trained on
   20 January–16 March, 7 normal weeks and 1 lockdown week, so lockdown traffic is still far from
   its training data. Input PSI cannot say "drift handled"; only the outcome layer can
   (WAPE 1.17 → 0.68). Follow-up: alert on weekly WAPE (layer 4) once labels arrive, and silence
   input-drift alerts in Alertmanager for a week after a drift-triggered retrain.
2. **v6 is better, not good.** WAPE 0.68 is still high: the 8-week window holds only one
   lockdown week. Recovery improves as more lockdown weeks arrive (weekly retrain, T17).
3. `for: 1m` and 5 s scrapes are demo values; production would use ~10 min to avoid paging on
   noise, which delays detection accordingly.

## Reproduce

```bash
uv run python -m src.registry rollback        # if a previous run left a retrained champion
docker compose up -d --build
uv run python -m src.simulate_incident
uv run python -m src.retrain --as-of 2020-03-23
curl.exe -X POST localhost:8000/admin/reload -H "X-Admin-Token: <ADMIN_TOKEN from .env>"
uv run python -m src.simulate_incident --phase recovery 2020-03-16 2020-03-23 --seconds 180 --out reports/recovery
```

Raw data (not committed, regenerated by the commands above): `reports/incident_timeline.csv`,
`reports/incident_summary*.json`, `reports/recovery/`, `reports/retrain_report.json`.
