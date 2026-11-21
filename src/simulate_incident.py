"""Incident simulator (T19): replay real taxi traffic against the API, first normal, then drifted.

Each request is a real (zone, hour) from demand.parquet: `recent_counts` are the 168 hours before
it, and the true pickup count is kept aside, as a label that arrives later. The API never sees it.

Default story (both windows are outside the training period, Jan-Feb 2020):
  normal  2-9 March 2020    demand as usual: no drift (T11 report: lag_1 PSI 0.002, WAPE 0.12)
  drift  16-23 March 2020   lockdown: demand collapses (lag_1 PSI > 0.25, WAPE 1.17)

Run, with the API up (uv run uvicorn api.app:app --port 8000):
  uv run python -m src.simulate_incident                          # 120 s normal, then 180 s drift
  uv run python -m src.simulate_incident --seconds 30 --rate 10    # quick try
  uv run python -m src.simulate_incident --phase drift 2020-04-06 2020-04-13   # one custom phase

Writes reports/incident_timeline.csv (one line per request: wall time, phase, request_id,
status, model_version, predicted, actual) and reports/incident_summary.json (per phase and per
model version). Together with the alert times, they are the timeline of the incident report.
The API keeps serving during /admin/reload, and a refused connection is logged, not fatal,
so the simulator can keep running through a rollback or a restart.
"""
import argparse
import csv
import json
import logging
import time
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DEMAND_PATH, ROOT, load_params

log = logging.getLogger(__name__)

REPORTS = ROOT / "reports"
DEFAULT_PHASES = [("normal", "2020-03-02", "2020-03-09"), ("drift", "2020-03-16", "2020-03-23")]
TIMELINE_FIELDS = ["sent_at", "phase", "request_id", "status", "model_version", "zone_id",
                   "target_hour", "predicted", "actual", "latency_ms"]

# send(payload, headers) -> (status_code, json body or None)
Sender = Callable[[dict, dict], tuple[int, dict | None]]


# ---------- requests from real data ----------

class History:
    """Hourly counts per zone, to build API requests and look up the true count."""

    def __init__(self, demand: pd.DataFrame, history_hours: int):
        self.history_hours = history_hours
        demand = demand.sort_values(["zone_id", "pickup_hour"])
        self.start = demand["pickup_hour"].min()
        self.counts = {int(z): g["target"].to_numpy(dtype=float) for z, g in demand.groupby("zone_id")}

    def request(self, zone_id: int, hour: pd.Timestamp) -> tuple[dict, float]:
        """The API payload for (zone, hour), and the true pickup count for that hour."""
        i = int((hour - self.start) / pd.Timedelta(hours=1))
        if i < self.history_hours:
            raise ValueError(f"{hour} has less than {self.history_hours} hours of history")
        series = self.counts[zone_id]
        payload = {"zone_id": zone_id, "target_hour": hour.isoformat(),
                   "recent_counts": series[i - self.history_hours:i].tolist()}
        return payload, float(series[i])

    def window(self, start, end, seed: int | None = None) -> Iterator[tuple[dict, float]]:
        """Every (zone, hour) of [start, end). In time order with seed=None, else shuffled.

        Shuffled is the default for the demo: a short phase in time order would only replay the
        first night hours (low demand), and a sliding-window PSI would fire during "normal".
        Shuffled, every slice of the phase is a fair sample of the whole week.
        """
        pairs = [(zone, hour) for hour in pd.date_range(start, end, freq="h", inclusive="left")
                 for zone in self.counts]
        if seed is not None:
            order = np.random.default_rng(seed).permutation(len(pairs))
            pairs = [pairs[i] for i in order]
        for zone, hour in pairs:
            yield self.request(zone, hour)


# ---------- sending ----------

@dataclass
class PhaseStats:
    sent: int = 0
    errors: int = 0
    predicted: float = 0.0
    actual: float = 0.0
    abs_error: float = 0.0
    versions: dict = field(default_factory=dict)

    def add(self, status: int, body: dict | None, actual: float) -> None:
        self.sent += 1
        if status != 200 or body is None:
            self.errors += 1
            return
        self.predicted += body["predicted_pickups"]
        self.actual += actual
        self.abs_error += abs(body["predicted_pickups"] - actual)
        v = str(body.get("model_version"))
        self.versions[v] = self.versions.get(v, 0) + 1

    def summary(self) -> dict:
        ok = self.sent - self.errors
        return {"sent": self.sent, "errors": self.errors,
                "mean_predicted": round(self.predicted / ok, 2) if ok else None,
                "mean_actual": round(self.actual / ok, 2) if ok else None,
                "wape": round(self.abs_error / self.actual, 4) if self.actual else None,
                "model_versions": self.versions}


def http_sender(url: str, timeout: float = 5.0) -> Sender:
    import requests  # only the live run needs it

    session = requests.Session()

    def send(payload: dict, headers: dict) -> tuple[int, dict | None]:
        try:
            r = session.post(f"{url.rstrip('/')}/predict", json=payload, headers=headers, timeout=timeout)
        except requests.RequestException as exc:  # API restarting or down: log it and keep going
            log.warning("request failed: %s", exc.__class__.__name__)
            return 0, None
        return r.status_code, (r.json() if r.status_code == 200 else None)

    return send


def run_phase(name: str, requests_iter: Iterator[tuple[dict, float]], send: Sender, seconds: float,
              rate: float, writer=None, clock=time.monotonic, sleep=time.sleep) -> PhaseStats:
    """Send requests at `rate` per second for `seconds`. Loops over the window if it runs out."""
    stats, items = PhaseStats(), list(requests_iter)
    if not items:
        raise ValueError(f"phase {name}: empty window")
    start, last_log, i = clock(), clock(), 0
    while clock() - start < seconds:
        payload, actual = items[i % len(items)]
        request_id = uuid.uuid4().hex
        t0 = clock()
        status, body = send(payload, {"X-Request-ID": request_id})
        latency = (clock() - t0) * 1000
        stats.add(status, body, actual)
        if writer is not None:
            writer.writerow({
                "sent_at": datetime.now().isoformat(timespec="milliseconds"), "phase": name,
                "request_id": request_id, "status": status,
                "model_version": body.get("model_version") if body else "",
                "zone_id": payload["zone_id"], "target_hour": payload["target_hour"],
                "predicted": body["predicted_pickups"] if body else "", "actual": actual,
                "latency_ms": round(latency, 1),
            })
        i += 1
        if clock() - last_log >= 10:
            log.info("[%s] %s", name, stats.summary())
            last_log = clock()
        pause = 1 / rate - (clock() - t0)
        if pause > 0:
            sleep(pause)
    return stats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Replay normal then drifted taxi traffic against the API")
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--rate", type=float, default=20, help="requests per second (default 20)")
    ap.add_argument("--seconds", type=float, nargs="+", default=[120, 180],
                    help="duration of each phase; one value applies to every phase (default 120 180)")
    ap.add_argument("--phase", nargs=3, action="append", metavar=("NAME", "START", "END"),
                    help="a phase replaying [START, END); repeat for several (default: normal, then drift)")
    ap.add_argument("--in-order", action="store_true",
                    help="replay hours in time order instead of shuffled (seed 42)")
    ap.add_argument("--out", type=Path, default=REPORTS)
    args = ap.parse_args(argv)

    phases = [tuple(p) for p in args.phase] if args.phase else DEFAULT_PHASES
    durations = args.seconds * len(phases) if len(args.seconds) == 1 else args.seconds
    if len(durations) != len(phases):
        ap.error(f"{len(phases)} phases but {len(args.seconds)} durations")

    history = History(pd.read_parquet(DEMAND_PATH), max(load_params()["features"]["lags"]))
    send = http_sender(args.url)
    args.out.mkdir(parents=True, exist_ok=True)
    summary = {"url": args.url, "rate": args.rate, "phases": []}

    with open(args.out / "incident_timeline.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TIMELINE_FIELDS)
        writer.writeheader()
        for (name, start, end), seconds in zip(phases, durations):
            began = datetime.now().isoformat(timespec="seconds")
            log.info("phase %s: replaying %s..%s for %ss at %s req/s", name, start, end, seconds, args.rate)
            seed = None if args.in_order else 42
            stats = run_phase(name, history.window(start, end, seed), send, seconds, args.rate, writer)
            summary["phases"].append({"name": name, "window": [start, end], "started": began,
                                      "ended": datetime.now().isoformat(timespec="seconds"),
                                      **stats.summary()})
            log.info("phase %s done: %s", name, stats.summary())
            f.flush()

    (args.out / "incident_summary.json").write_text(json.dumps(summary, indent=2) + "\n", newline="\n")
    log.info("timeline: %s, summary: %s", args.out / "incident_timeline.csv", args.out / "incident_summary.json")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raise SystemExit(main())

