"""Rollback logic tests (task T18).

The registry helpers (promote / rollback / champion_version) only ever call a small set
of MlflowClient methods. We drive them with a FakeClient that records aliases and tags in
dicts, so these tests prove the alias moves correctly with NO MLflow server running.

Run: uv run pytest tests/test_registry.py
"""
import pytest
from mlflow.exceptions import MlflowException

from src import registry

NAME = "taxi-demand-forecaster"


class FakeMV:
    def __init__(self, version, tags):
        self.version = version
        self.tags = tags


class FakeClient:
    """Minimal stand-in for MlflowClient: aliases and tags kept in memory."""

    def __init__(self):
        self.aliases: dict[str, str] = {}        # alias -> version
        self.tags: dict[str, dict] = {}          # version -> {key: value}

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


def test_first_promote_sets_champion_no_replaces():
    c = FakeClient()
    registry.promote(c, NAME, "1")
    assert registry.champion_version(c, NAME) == "1"
    assert "replaces" not in c.tags.get("1", {})  # nothing to replace yet


def test_promote_records_replaces():
    c = FakeClient()
    registry.promote(c, NAME, "1")
    registry.promote(c, NAME, "2")                # v2 takes over
    assert registry.champion_version(c, NAME) == "2"
    assert c.tags["2"]["replaces"] == "1"         # knows where to go back


def test_rollback_restores_previous_champion():
    c = FakeClient()
    registry.promote(c, NAME, "1")
    registry.promote(c, NAME, "2")
    previous = registry.rollback(c, NAME)
    assert previous == "1"
    assert registry.champion_version(c, NAME) == "1"   # alias moved back
    assert c.tags["2"]["rolled_back"] == "true"        # bad version marked


def test_rollback_without_history_raises():
    c = FakeClient()
    registry.promote(c, NAME, "1")                # only one version, no replaces tag
    with pytest.raises(RuntimeError):
        registry.rollback(c, NAME)
