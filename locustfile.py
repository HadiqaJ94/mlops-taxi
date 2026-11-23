"""Load test for the taxi-demand API (task T14).

Ramp the number of users in the Locust UI and watch the 95th-percentile latency.
Report the throughput (requests/second) at the last point P95 stays under the budget
of 200 ms - that is the "knee", just before requests start to queue.

Run (API up via `docker compose up -d --build`):
    uv run locust -f locustfile.py --host http://localhost:8000
    # then open http://localhost:8089 in the browser

Record one run for the README (use the user count you found at the knee):
    uv run locust -f locustfile.py --host http://localhost:8000 \
        --headless -u 50 -r 10 -t 60s --csv t14_results
"""
import json
import pathlib
import random

from locust import HttpUser, between, task

# Only send zones the model actually serves, or /predict returns 422 and the numbers mean nothing.
_zones_file = pathlib.Path("data/processed/zones.json")
ZONES = (
    json.loads(_zones_file.read_text())
    if _zones_file.exists()
    else [161, 237, 236, 142, 48, 230, 163, 170]
)

# Pre-build one realistic payload per zone, ONCE. 168 hourly counts = one week (the longest lag).
# Building them upfront keeps the load generator light, so we measure the API, not the client.
_rng = random.Random(42)
PAYLOADS = [
    {
        "zone_id": int(z),
        "target_hour": "2020-02-20T18:00:00",   # on the hour (the API requires it)
        "recent_counts": [float(_rng.randint(0, 80)) for _ in range(168)],
    }
    for z in ZONES
]


class TaxiUser(HttpUser):
    # Minimal think time: you control the load with the user count in the UI, not by waiting.
    wait_time = between(0.0, 0.05)

    @task
    def predict(self):
        self.client.post("/predict", json=random.choice(PAYLOADS))
