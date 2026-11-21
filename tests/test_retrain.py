"""Retraining tests (T16), on a small synthetic grid, with MLflow and the drift check mocked."""
import copy

import numpy as np
import pandas as pd
import pytest

from src import gate, retrain
from src.config import load_params
from src.features import build_features

PARAMS = load_params()
LAGS = PARAMS["features"]["lags"]


@pytest.fixture(scope="module")
def feats():
    """2 zones, hourly, Jan to March 2020; demand halves from March."""
    rng = np.random.default_rng(0)
    hours = pd.date_range("2020-01-01", "2020-03-31 23:00", freq="h")
    level = np.where(hours >= "2020-03-01", 50, 100)
    frames = [pd.DataFrame({"zone_id": z, "pickup_hour": hours, "target": rng.poisson(level),
                            "period": np.where(hours >= "2020-03-01", "drift", "reference")})
              for z in (1, 2)]
    return build_features(pd.concat(frames, ignore_index=True), LAGS)


@pytest.fixture
def params():
    p = copy.deepcopy(PARAMS)
    p["train"]["max_iter"] = 20          # fast
    p["retrain"].update(train_weeks=4, val_weeks=1, as_of=None)
    return p


@pytest.fixture
def mocks(monkeypatch):
    """Record calls instead of touching MLflow; the drift check returns `state['drift']`."""
    calls = {"register": [], "gate": [], "train": []}
    state = {"drift": True}

    def fake_drift(start, end, out=None):
        return {"drift": state["drift"], "reasons": ["input drift on lag_1"] if state["drift"] else [],
                "features": {"lag_1": {"status": "drift" if state["drift"] else "ok"},
                             "mean_fare": {"status": "ok"}}}

    real_train = retrain.train_model

    def spy_train(f, start, end, p):
        calls["train"].append((pd.Timestamp(start), pd.Timestamp(end)))
        return real_train(f, start, end, p)

    def fake_register(model, p, metrics, example, tags, extra_params=None):
        calls["register"].append(tags)
        return "run123", "7"

    def fake_gate(val_start, val_end, min_improvement=None, dry_run=False, out=None, feats=None):
        calls["gate"].append((pd.Timestamp(val_start), pd.Timestamp(val_end)))
        return {"decision": "promoted", "promote": True, "reason": "test", "champion_version": "1",
                "wape": {"candidate": 0.1, "baseline": 0.2, "champion": 0.3}}

    monkeypatch.setattr(retrain, "check_drift", fake_drift)
    monkeypatch.setattr(retrain, "train_model", spy_train)
    monkeypatch.setattr(retrain, "log_and_register", fake_register)
    monkeypatch.setattr(gate, "run", fake_gate)
    return calls, state


# ---------- windows ----------

def test_windows_are_consecutive_and_do_not_overlap():
    w = retrain.compute_windows("2020-03-23", 8, 1, "2020-01-08")
    assert w["val_end"] == pd.Timestamp("2020-03-23")
    assert w["val_start"] == pd.Timestamp("2020-03-16")
    assert w["train_end"] == w["val_start"]                    # end excluded: no shared hour
    assert w["train_start"] == pd.Timestamp("2020-01-20")      # 8 weeks before validation


def test_windows_need_enough_history():
    with pytest.raises(ValueError):
        retrain.compute_windows("2020-02-01", 8, 1, "2020-01-08")
    with pytest.raises(ValueError):
        retrain.compute_windows("2020-03-23 12:30", 8, 1, "2020-01-08")   # not on the hour


# ---------- the trigger ----------

def test_no_drift_means_no_retrain(feats, params, mocks):
    calls, state = mocks
    state["drift"] = False
    report = retrain.run_retrain(feats, params)
    assert report["retrained"] is False and "no retrain needed" in report["reason"]
    assert calls["train"] == [] and calls["register"] == [] and calls["gate"] == []


def test_drift_retrains_registers_and_calls_the_gate_on_the_recent_week(feats, params, mocks):
    calls, _ = mocks
    report = retrain.run_retrain(feats, params, as_of="2020-03-30")
    assert report["retrained"] is True and report["trigger"] == "drift"
    assert report["drift"]["features_in_drift"] == ["lag_1"]

    (train_start, train_end), = calls["train"]
    (val_start, val_end), = calls["gate"]
    assert (val_start, val_end) == (pd.Timestamp("2020-03-23"), pd.Timestamp("2020-03-30"))
    assert train_end == val_start and train_start == pd.Timestamp("2020-02-24")
    assert calls["register"][0]["trigger"] == "drift"
    assert report["candidate"]["version"] == "7" and report["gate"]["decision"] == "promoted"


def test_force_retrains_without_drift(feats, params, mocks):
    calls, state = mocks
    state["drift"] = False
    report = retrain.run_retrain(feats, params, force=True)
    assert report["retrained"] is True and report["trigger"] == "manual"
    assert calls["register"][0]["trigger"] == "manual"
    assert report["windows"]["validation"][1] == "2020-04-01 00:00:00"   # default as_of = end of data


def test_training_rows_are_all_before_validation(feats, params, monkeypatch):
    """No leakage: the frame handed to fit() ends before the validation window starts."""
    seen = {}

    class Spy:
        def fit(self, X, y):
            seen["n"] = len(X)
            return self

    monkeypatch.setattr("src.train.make_model", lambda tp: Spy())
    from src.train import train_model, window
    train_model(feats, "2020-02-24", "2020-03-23", params)
    rows = window(feats, "2020-02-24", "2020-03-23")
    assert seen["n"] == len(rows)
    assert rows["pickup_hour"].max() < pd.Timestamp("2020-03-23")


def test_dry_run_registers_and_promotes_nothing(feats, params, mocks, monkeypatch):
    calls, _ = mocks
    monkeypatch.setattr(retrain, "load_champion_model", lambda: (None, None))
    report = retrain.run_retrain(feats, params, dry_run=True)
    assert calls["register"] == [] and calls["gate"] == []
    assert report["gate"]["dry_run"] is True and report["candidate"]["version"] is None
    assert report["gate"]["decision"] in ("promoted", "rejected")


def test_main_exit_code_on_error(monkeypatch, tmp_path):
    def boom(*a, **k):
        raise ValueError("bad window")
    monkeypatch.setattr(retrain.pd, "read_parquet", lambda path: None)   # no data file needed
    monkeypatch.setattr(retrain, "build_features", lambda df, lags: None)
    monkeypatch.setattr(retrain, "run_retrain", boom)
    assert retrain.main(["--out", str(tmp_path / "r.json")]) == 1
