"""Training gate tests (T6), with no MLflow server.

decide() is pure. run_gate() is driven with an in-memory FakeClient (same idea as
tests/test_registry.py) and stub models, on a small synthetic zone x hour grid.
"""
import numpy as np
import pandas as pd
import pytest
from mlflow.exceptions import MlflowException

from src import gate, registry
from src.config import load_params
from src.features import build_features
from src.train import time_split

NAME = "taxi-demand-forecaster"
LAGS = load_params()["features"]["lags"]


# ---------- decide() ----------

def test_beats_baseline_and_champion_is_promoted():
    promote, reason = gate.decide(0.10, 0.18, 0.12, 0.01)
    assert promote and "beats" in reason


def test_beats_baseline_but_not_champion_is_rejected():
    promote, reason = gate.decide(0.11, 0.18, 0.10, 0.01)
    assert not promote and "champion" in reason


def test_worse_than_baseline_is_rejected_even_without_champion():
    assert gate.decide(0.20, 0.18, None, 0.01)[0] is False
    assert gate.decide(0.20, 0.18, 0.30, 0.01)[0] is False   # beating a bad champion is not enough


def test_no_champion_and_beats_baseline_is_promoted():
    promote, reason = gate.decide(0.10, 0.18, None, 0.01)
    assert promote and "no champion" in reason


def test_tie_with_champion_depends_on_min_improvement():
    assert gate.decide(0.10, 0.18, 0.10, 0.0)[0] is True        # 0 = better or equal
    assert gate.decide(0.10, 0.18, 0.10, 0.01)[0] is False      # 1% needed, 0% gained
    assert gate.decide(0.099, 0.18, 0.10, 0.01)[0] is True      # exactly 1% better


def test_tie_with_baseline_is_rejected():
    assert gate.decide(0.18, 0.18, None, 0.0)[0] is False


def test_min_improvement_out_of_range():
    with pytest.raises(ValueError):
        gate.decide(0.1, 0.2, 0.1, 1.5)


# ---------- the fixed window ----------

def test_gate_window_is_the_train_validation_window():
    """params.yaml gate == the last train.val_days days of the reference period (what drift.py uses)."""
    params = load_params()
    hours = pd.date_range("2020-01-01", "2020-02-29 23:00", freq="h")   # the reference months
    ref = pd.DataFrame({"pickup_hour": hours})
    _, val = time_split(ref, params["train"]["val_days"])
    assert val["pickup_hour"].min() == pd.Timestamp(params["gate"]["val_start"])
    assert val["pickup_hour"].max() + pd.Timedelta(hours=1) == pd.Timestamp(params["gate"]["val_end"])


# ---------- run_gate() with a fake registry ----------

class FakeMV:
    def __init__(self, version, tags):
        self.version, self.tags = version, tags


class FakeClient:
    def __init__(self, aliases=None):
        self.aliases: dict[str, str] = dict(aliases or {})
        self.tags: dict[str, dict] = {}

    def get_model_version_by_alias(self, name, alias):
        if alias not in self.aliases:
            raise MlflowException(f"alias {alias} not set")
        v = self.aliases[alias]
        return FakeMV(v, self.tags.get(v, {}))

    def set_registered_model_alias(self, name, alias, version):
        self.aliases[alias] = str(version)

    def set_model_version_tag(self, name, version, key, value):
        self.tags.setdefault(str(version), {})[key] = value

    def get_model_version(self, name, version):
        return FakeMV(str(version), self.tags.get(str(version), {}))


class ScaledLag:
    """Stub model: predicts lag_1 * factor. factor 1 is decent, far from 1 is bad."""

    def __init__(self, factor):
        self.factor = factor

    def predict(self, X):
        return X["lag_1"].to_numpy(dtype=float) * self.factor


@pytest.fixture(scope="module")
def feats():
    rng = np.random.default_rng(0)
    hours = pd.date_range("2020-01-01", "2020-02-29 23:00", freq="h")
    # steady upward trend: last hour (lag_1) is a good guess, same hour last week (lag_168) lags behind
    level = 20 + 0.3 * np.arange(len(hours))
    frames = [pd.DataFrame({"zone_id": z, "pickup_hour": hours,
                            "target": rng.poisson(level), "period": "reference"}) for z in (1, 2)]
    return build_features(pd.concat(frames, ignore_index=True), LAGS)


def run(client, models, feats, **kw):
    return gate.run_gate(client, NAME, feats, "2020-02-16", "2020-03-01", LAGS, kw.pop("min_improvement", 0.01),
                         load_model=lambda name, v: models[v], **kw)


def test_run_gate_promotes_a_better_candidate(feats):
    client = FakeClient({"candidate": "2", "champion": "1"})
    report = run(client, {"1": ScaledLag(1.5), "2": ScaledLag(1.0)}, feats)
    assert report["decision"] == "promoted"
    assert registry.champion_version(client, NAME) == "2"
    assert client.tags["2"]["replaces"] == "1"            # promoted through registry.promote (T18)
    assert client.tags["2"]["gate_decision"] == "promoted"
    assert report["wape"]["candidate"] < report["wape"]["champion"]


def test_run_gate_rejects_a_worse_candidate(feats):
    client = FakeClient({"candidate": "2", "champion": "1"})
    report = run(client, {"1": ScaledLag(1.0), "2": ScaledLag(1.5)}, feats)
    assert report["decision"] == "rejected"
    assert registry.champion_version(client, NAME) == "1"   # champion untouched
    assert client.tags["2"]["gate_decision"] == "rejected"
    assert "gate_reason" in client.tags["2"]


def test_run_gate_first_model_becomes_champion(feats):
    client = FakeClient({"candidate": "1"})
    report = run(client, {"1": ScaledLag(1.0)}, feats)
    assert report["decision"] == "promoted" and report["champion_version"] is None
    assert registry.champion_version(client, NAME) == "1"


def test_run_gate_dry_run_changes_nothing(feats):
    client = FakeClient({"candidate": "2", "champion": "1"})
    report = run(client, {"1": ScaledLag(1.5), "2": ScaledLag(1.0)}, feats, dry_run=True)
    assert report["decision"] == "promoted" and report["dry_run"] is True
    assert registry.champion_version(client, NAME) == "1"
    assert client.tags == {}


def test_run_gate_skips_when_candidate_is_already_champion(feats):
    client = FakeClient({"candidate": "1", "champion": "1"})
    report = run(client, {}, feats)
    assert report["decision"] == "skipped"
    assert client.tags == {}


def test_run_gate_without_candidate_raises(feats):
    with pytest.raises(MlflowException):
        run(FakeClient({"champion": "1"}), {}, feats)


def test_run_gate_empty_window_raises(feats):
    client = FakeClient({"candidate": "1"})
    with pytest.raises(ValueError):
        gate.run_gate(client, NAME, feats, "2021-01-01", "2021-01-08", LAGS, 0.01,
                      load_model=lambda name, v: ScaledLag(1.0))


def test_main_exit_codes(monkeypatch):
    monkeypatch.setattr(gate, "run", lambda *a, **k: {"decision": "rejected"})
    assert gate.main([]) == 0                                    # a rejection is not an error

    def boom(*a, **k):
        raise MlflowException("no candidate")
    monkeypatch.setattr(gate, "run", boom)
    assert gate.main([]) == 1
