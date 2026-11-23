"""API tests for the serving service (task T8).

The app loads the champion model from MLflow at startup. Tests must not depend on a running
MLflow, so the `client` fixture swaps the loader (api.app.load_champion) for a tiny stub
model. The stub returns lag_1 (the most recent hourly count), so predictions are predictable
and we can assert on them.

Run: uv run pytest tests/test_api.py
"""
import pytest
from fastapi.testclient import TestClient

import api.app as app_module

MIN_HISTORY = app_module.MIN_HISTORY       # 168 -> one week of hourly history
STUB_ZONES = frozenset({50, 161, 237})     # the zones the stub model "knows"


class StubModel:
    """Stand-in for the real model: predicts the most recent hour's count (lag_1)."""

    def predict(self, features):
        return features["lag_1"].to_numpy()


@pytest.fixture
def client(monkeypatch):
    """A TestClient whose startup loads the stub instead of the real champion."""

    def fake_load():
        return {"model": StubModel(), "version": "test", "zones": STUB_ZONES}

    monkeypatch.setattr(app_module, "load_champion", fake_load)
    with TestClient(app_module.app) as c:  # `with` runs the lifespan (startup)
        yield c


def make_request(zone_id=237, hour="2020-02-20T18:00:00", counts=None):
    if counts is None:
        counts = [float(i) for i in range(MIN_HISTORY)]  # 0.0 .. 167.0
    return {"zone_id": zone_id, "target_hour": hour, "recent_counts": counts}


def test_predict_valid_request(client):
    """A well-formed request returns 200 and a complete prediction."""
    counts = [float(i) for i in range(MIN_HISTORY)]
    resp = client.post("/predict", json=make_request(counts=counts))
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_version"] == "test"
    assert "model_ms" in body
    assert body["predicted_pickups"] == counts[-1]  # stub returns lag_1 = the most recent count


def test_bad_input_returns_422(client):
    """Too few recent_counts is rejected with 422, and the error names the field."""
    resp = client.post("/predict", json=make_request(counts=[1.0, 2.0, 3.0]))
    assert resp.status_code == 422
    assert "recent_counts" in resp.text


def test_unknown_zone_returns_422(client):
    """A zone the model was not trained on is refused with 422."""
    resp = client.post("/predict", json=make_request(zone_id=100))  # in range, not in STUB_ZONES
    assert resp.status_code == 422
    assert "not served" in resp.text


def test_health(client):
    """/health reports status ok and the loaded model version."""
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_version"] == "test"


def test_batch_equals_single(client):
    """Scoring through /predict_batch matches scoring each item via /predict."""
    items = [
        make_request(zone_id=161, counts=[float(i) for i in range(MIN_HISTORY)]),
        make_request(zone_id=50, hour="2020-02-16T09:00:00",
                     counts=[float((i * 2) % 7) for i in range(MIN_HISTORY)]),
    ]
    singles = [client.post("/predict", json=it).json()["predicted_pickups"] for it in items]
    batch = client.post("/predict_batch", json={"items": items}).json()["predictions"]
    assert [p["predicted_pickups"] for p in batch] == singles


def test_admin_reload_refuses_the_default_token(client, monkeypatch):
    """The .env.example value is public: with it, admin routes stay disabled (T20 finding)."""
    monkeypatch.setenv("ADMIN_TOKEN", "change-me")
    resp = client.post("/admin/reload", headers={"X-Admin-Token": "change-me"})
    assert resp.status_code == 503


def test_admin_reload_requires_token(client, monkeypatch):
    """With ADMIN_TOKEN set, /admin/reload without the header is rejected with 401."""
    monkeypatch.setenv("ADMIN_TOKEN", "secret")
    resp = client.post("/admin/reload")
    assert resp.status_code == 401


@pytest.fixture
def degraded_client(monkeypatch):
    """A TestClient whose startup FAILS to load a model, so the service runs degraded."""

    def fail_load():
        raise RuntimeError("no model available")

    monkeypatch.setattr(app_module, "load_champion", fail_load)
    app_module.state["current"] = None  # ensure degraded even if a prior test left a model loaded
    with TestClient(app_module.app) as c:
        yield c


def test_no_model_serves_503(degraded_client):
    """With no model loaded the service stays up: /health and /predict answer 503, not a crash."""
    health = degraded_client.get("/health")
    assert health.status_code == 503
    assert health.json()["status"] == "degraded"
    assert degraded_client.post("/predict", json=make_request()).status_code == 503


def test_extra_field_returns_422(client):
    """An unexpected field (e.g. a typo) is rejected with 422, not silently ignored (extra='forbid')."""
    payload = make_request()
    payload["surprise"] = 1
    assert client.post("/predict", json=payload).status_code == 422


def test_reload_swaps_model_version(client, monkeypatch):
    """/admin/reload changes which version the live service serves - the rollback mechanism (T18)."""
    assert client.get("/health").json()["model_version"] == "test"
    monkeypatch.setenv("ADMIN_TOKEN", "secret")
    monkeypatch.setattr(
        app_module,
        "load_champion",
        lambda: {"model": StubModel(), "version": "test2", "zones": STUB_ZONES},
    )
    resp = client.post("/admin/reload", headers={"X-Admin-Token": "secret"})
    assert resp.status_code == 200
    assert resp.json()["model_version"] == "test2"
    assert client.get("/health").json()["model_version"] == "test2"
