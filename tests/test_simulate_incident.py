"""Tests for src/simulate_incident.py (T19): requests built from real-shaped data, pacing,
statistics, and one run through the real FastAPI app with a stub model (no server, no MLflow).
"""
import csv
import io

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import api.app as app_module
from src.simulate_incident import TIMELINE_FIELDS, History, run_phase

HISTORY = 168
ZONES = (13, 237)


@pytest.fixture
def demand():
    hours = pd.date_range("2020-01-01", periods=24 * 10, freq="h")
    return pd.concat([
        pd.DataFrame({"zone_id": z, "pickup_hour": hours, "target": np.arange(len(hours)) + 1000 * z})
        for z in ZONES
    ], ignore_index=True)


class FakeClock:
    """time.monotonic and time.sleep that move a counter instead of waiting."""

    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, seconds):
        self.t += seconds


def test_request_uses_the_168_hours_before_and_keeps_the_label_aside(demand):
    history = History(demand, HISTORY)
    hour = pd.Timestamp("2020-01-01") + pd.Timedelta(hours=200)
    payload, actual = history.request(13, hour)
    assert len(payload["recent_counts"]) == HISTORY
    assert payload["recent_counts"][-1] == 13000 + 199  # the hour just before target_hour
    assert actual == 13000 + 200                         # the true count, never sent
    assert set(payload) == {"zone_id", "target_hour", "recent_counts"}


def test_request_without_a_week_of_history_is_refused(demand):
    with pytest.raises(ValueError):
        History(demand, HISTORY).request(13, pd.Timestamp("2020-01-02"))


def test_window_goes_hour_by_hour_over_all_zones(demand):
    items = list(History(demand, HISTORY).window("2020-01-08", "2020-01-08 03:00"))
    assert len(items) == 3 * len(ZONES)
    assert [p["zone_id"] for p, _ in items[:2]] == list(ZONES)


def test_shuffled_window_is_the_same_requests_in_a_fixed_random_order(demand):
    history = History(demand, HISTORY)
    ordered = list(history.window("2020-01-08", "2020-01-09"))
    shuffled = list(history.window("2020-01-08", "2020-01-09", seed=42))
    key = lambda item: (item[0]["zone_id"], item[0]["target_hour"])
    assert sorted(map(key, shuffled)) == sorted(map(key, ordered))
    assert list(map(key, shuffled)) != list(map(key, ordered))
    assert list(map(key, shuffled)) == list(map(key, history.window("2020-01-08", "2020-01-09", seed=42)))


def test_run_phase_paces_counts_and_logs(demand):
    clock = FakeClock()
    items = list(History(demand, HISTORY).window("2020-01-08", "2020-01-08 02:00"))

    def send(payload, headers):
        clock.t += 0.01  # each call takes 10 ms
        assert headers["X-Request-ID"]
        if payload["zone_id"] == 237:
            return 503, None
        return 200, {"predicted_pickups": 2 * payload["recent_counts"][-1], "model_version": "7"}

    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=TIMELINE_FIELDS)
    stats = run_phase("normal", iter(items), send, seconds=2, rate=10, writer=writer,
                      clock=clock, sleep=clock.sleep)

    assert stats.sent == 20                 # 2 s at 10 req/s, looping over the 4 requests
    assert stats.errors == 10               # zone 237 answered 503
    assert stats.summary()["model_versions"] == {"7": 10}
    rows = list(csv.DictReader(io.StringIO(out.getvalue()), fieldnames=TIMELINE_FIELDS))
    assert len(rows) == 20 and rows[0]["phase"] == "normal"


def test_run_phase_against_the_real_app(demand, monkeypatch):
    """End to end through FastAPI: validation, model_version, predictions."""

    class LastHour:
        def predict(self, features):
            return features["lag_1"].to_numpy()

    monkeypatch.setattr(app_module, "load_champion",
                        lambda: {"model": LastHour(), "version": "test", "zones": frozenset(ZONES)})
    clock = FakeClock()
    items = list(History(demand, HISTORY).window("2020-01-08", "2020-01-08 05:00"))
    with TestClient(app_module.app) as client:
        def send(payload, headers):
            clock.t += 0.01
            r = client.post("/predict", json=payload, headers=headers)
            assert r.headers["X-Request-ID"] == headers["X-Request-ID"]  # the id joins logs and labels
            return r.status_code, r.json()

        stats = run_phase("drift", iter(items), send, seconds=1, rate=10, clock=clock, sleep=clock.sleep)

    summary = stats.summary()
    assert summary["errors"] == 0
    assert summary["model_versions"] == {"test": stats.sent}
    assert summary["wape"] < 0.01  # the stub predicts the previous hour of a slowly rising series
