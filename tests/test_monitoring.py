"""Monitoring tests (T10). No MLflow, no Docker: the stub model of test_api.py is reused."""
import json
import logging
import math

import numpy as np
import pandas as pd
from prometheus_client import REGISTRY

from api import monitoring
from tests.test_api import MIN_HISTORY, client, make_request  # noqa: F401  (client is a fixture)


def sample(name, labels=None):
    return REGISTRY.get_sample_value(name, labels or {})


def test_metrics_endpoint_exposes_latency_and_version_counter(client):  # noqa: F811
    before = sample("taxi_predictions_total", {"model_version": "test"}) or 0
    assert client.post("/predict", json=make_request()).status_code == 200
    assert client.post("/predict_batch", json={"items": [make_request(), make_request(zone_id=50)]}).status_code == 200

    body = client.get("/metrics").text
    assert "taxi_request_seconds_bucket" in body
    assert sample("taxi_predictions_total", {"model_version": "test"}) == before + 3
    assert sample("taxi_model_loaded") == 1


def test_error_status_is_recorded_and_metrics_route_is_not(client):  # noqa: F811
    client.post("/predict", json=make_request(counts=[1.0]))  # 422
    client.get("/metrics")
    assert sample("taxi_request_seconds_count", {"endpoint": "/predict", "status": "422"}) >= 1
    assert sample("taxi_request_seconds_count", {"endpoint": "/metrics", "status": "200"}) is None


def test_unknown_paths_share_one_label(client):  # noqa: F811
    client.get("/wp-admin")
    client.get("/.env")
    assert sample("taxi_request_seconds_count", {"endpoint": "other", "status": "404"}) >= 2


def make_monitor(window=300, every=20):
    """A monitor whose reference is lag_1 ~ uniform(0, 100)."""
    rng = np.random.default_rng(0)
    ref = pd.DataFrame({"lag_1": rng.uniform(0, 100, 5000)})
    m = monitoring.Monitor(window=window, every=every)
    m.set_reference(ref, ["lag_1"], ["lag_1"], bins=10)
    return m, rng


def feed(m, values, version="t"):
    m.observe(pd.DataFrame({"lag_1": values}), np.asarray(values, dtype=float), version)


def test_psi_gauge_stays_low_on_normal_traffic_and_jumps_on_drift():
    m, rng = make_monitor()
    feed(m, rng.uniform(0, 100, 300))
    normal = sample("taxi_input_psi", {"feature": "lag_1"})
    assert normal < 0.1

    feed(m, rng.uniform(0, 100, 300) * 3)  # every value times 3: the window is now all drifted
    drifted = sample("taxi_input_psi", {"feature": "lag_1"})
    assert drifted > 0.25
    assert sample("taxi_prediction_mean") > 100


def test_no_psi_before_the_window_is_full():
    m, rng = make_monitor()
    feed(m, rng.uniform(0, 100, 150))  # half a window
    assert math.isnan(sample("taxi_input_psi", {"feature": "lag_1"}))


def test_model_swap_resets_the_prediction_window():
    table, rng = table_with_a_regime_change()
    m = monitoring.Monitor(window=300, every=20)
    m.set_features(table, ["lag_1"], ["lag_1"], bins=10)

    class Const:
        def predict(self, X):
            return np.full(len(X), 50.0)

    feed(m, rng.uniform(0, 100, 30))
    assert len(m.preds) == 30
    m.on_model_loaded(Const(), "2")
    assert len(m.preds) == 0
    assert sample("taxi_reference_prediction_mean") == 50.0
    assert sample("taxi_model_loaded") == 1


def test_json_log_carries_the_request_id():
    record = logging.LogRecord("api.access", logging.INFO, __file__, 1, "request", None, None)
    record.fields = {"path": "/predict", "status": 200}
    token = monitoring.request_id_var.set("abc123")
    try:
        line = monitoring.JsonFormatter().format(record)
    finally:
        monitoring.request_id_var.reset(token)
    out = json.loads(line)
    assert out["request_id"] == "abc123" and out["path"] == "/predict" and out["level"] == "INFO"


def test_drift_through_the_api_moves_the_gauge(client, monkeypatch):  # noqa: F811
    """End to end: requests -> predict_many -> observe -> /metrics."""
    m, rng = make_monitor()
    monkeypatch.setattr(monitoring, "monitor", m)

    def batch(scale):
        items = []
        for last in rng.uniform(0, 100, 300) * scale:
            counts = [10.0] * (MIN_HISTORY - 1) + [float(last)]  # lag_1 = last
            items.append(make_request(counts=counts))
        assert client.post("/predict_batch", json={"items": items}).status_code == 200

    batch(1)
    assert sample("taxi_input_psi", {"feature": "lag_1"}) < 0.1
    batch(3)
    assert sample("taxi_input_psi", {"feature": "lag_1"}) > 0.25


def test_gauges_are_nan_not_zero_when_there_is_no_data():
    """A fake 0 made PredictionMeanShift go pending at startup (|0 - ref| / ref = 1)."""
    make_monitor()
    for name in ("taxi_prediction_mean", "taxi_prediction_psi"):
        assert math.isnan(sample(name))
    assert math.isnan(sample("taxi_input_psi", {"feature": "lag_1"}))


class Const:
    def predict(self, X):
        return np.full(len(X), 50.0)


def table_with_a_regime_change():
    """Jan-Feb (reference period): lag_1 ~ U(0, 100). April: demand collapsed, lag_1 ~ U(0, 20)."""
    rng = np.random.default_rng(1)
    jan = pd.DataFrame({"lag_1": rng.uniform(0, 100, 1400), "period": "reference",   # Jan 1 to Feb 28
                        "pickup_hour": pd.date_range("2020-01-01", periods=1400, freq="h")})
    apr = pd.DataFrame({"lag_1": rng.uniform(0, 20, 2000), "period": "drift",
                        "pickup_hour": pd.date_range("2020-04-01", periods=2000, freq="h")})
    return pd.concat([jan, apr], ignore_index=True), rng


def test_reference_follows_the_retrained_model():
    """After a retrain on April, April traffic is normal again: the alert must be able to clear."""
    table, rng = table_with_a_regime_change()
    m = monitoring.Monitor(window=300, every=20)
    m.set_features(table, ["lag_1"], ["lag_1"], bins=10)

    m.on_model_loaded(Const(), "1")                       # first model: reference period
    feed(m, rng.uniform(0, 20, 300))                      # April-like traffic
    assert sample("taxi_input_psi", {"feature": "lag_1"}) > 0.25

    window = (pd.Timestamp("2020-04-01"), pd.Timestamp("2020-04-30"))
    m.on_model_loaded(Const(), "2", window)               # retrained on April
    assert math.isnan(sample("taxi_input_psi", {"feature": "lag_1"}))   # old alert cleared at once
    feed(m, rng.uniform(0, 20, 300))
    assert sample("taxi_input_psi", {"feature": "lag_1"}) < 0.1


def test_unknown_training_window_falls_back_to_the_reference_period():
    table, _ = table_with_a_regime_change()
    m = monitoring.Monitor(window=300, every=20)
    m.set_features(table, ["lag_1"], ["lag_1"], bins=10)
    m.on_model_loaded(Const(), "3", (pd.Timestamp("2030-01-01"), pd.Timestamp("2030-02-01")))
    assert len(m.reference) == 1400 and m.reference["lag_1"].max() > 90   # the Jan-Feb rows


class FakeClient:
    def __init__(self, tags=None, fail=False):
        self.tags, self.fail = tags or {}, fail

    def get_model_version(self, name, version):
        if self.fail:
            raise RuntimeError("mlflow down")
        return type("MV", (), {"run_id": "r1"})

    def get_run(self, run_id):
        return type("Run", (), {"data": type("D", (), {"tags": self.tags})})


def test_train_window_reads_the_run_tag():
    """The tag train.py already logs: f"{train_start}..{val_start}"."""
    tags = {"train_window": "2020-04-01 00:00:00..2020-04-29 00:00:00"}
    assert monitoring.train_window(FakeClient(tags), "m", "2") == (
        pd.Timestamp("2020-04-01"), pd.Timestamp("2020-04-29"))
    assert monitoring.train_window(FakeClient({}), "m", "1") is None       # a run without the tag
    assert monitoring.train_window(FakeClient(fail=True), "m", "1") is None  # MLflow down: no crash
