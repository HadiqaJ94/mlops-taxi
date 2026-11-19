"""MLflow Model Registry helpers: promote a version, roll back to the previous one.

Convention: the alias `champion` is what the API serves (models:/<name>@champion).
`candidate` is a freshly trained version waiting to be promoted.
Each promotion stores a `replaces` tag on the new version -> rollback knows where to go back to.
"""
import argparse
import os

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from src.config import load_params

CHAMPION, CANDIDATE = "champion", "candidate"


def get_client() -> MlflowClient:
    uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(uri)
    return MlflowClient()


def model_name() -> str:
    return load_params()["mlflow"]["registered_model"]


def champion_version(client: MlflowClient, name: str) -> str | None:
    try:
        return client.get_model_version_by_alias(name, CHAMPION).version
    except MlflowException:
        return None


def promote(client: MlflowClient, name: str, version: str) -> None:
    old = champion_version(client, name)
    if old is not None and str(old) == str(version):
        return
    if old is not None:
        client.set_model_version_tag(name, str(version), "replaces", str(old))
    client.set_registered_model_alias(name, CHAMPION, str(version))
    print(f"champion: v{old} -> v{version}")


def rollback(client: MlflowClient, name: str) -> str:
    current = champion_version(client, name)
    if current is None:
        raise RuntimeError("No champion to roll back from.")
    previous = client.get_model_version(name, current).tags.get("replaces")
    if previous is None:
        raise RuntimeError(f"v{current} has no 'replaces' tag: nothing to roll back to.")
    client.set_model_version_tag(name, current, "rolled_back", "true")
    client.set_registered_model_alias(name, CHAMPION, previous)
    print(f"ROLLBACK: champion v{current} -> v{previous}")
    return previous


def status(client: MlflowClient, name: str) -> None:
    versions = sorted(client.search_model_versions(f"name='{name}'"), key=lambda m: int(m.version))
    for mv in versions:
        mv = client.get_model_version(name, mv.version)  # search results omit aliases
        aliases = ",".join(mv.aliases) or "-"
        print(f"v{mv.version}  aliases={aliases}  tags={dict(mv.tags)}  run={mv.run_id[:8]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Model registry operations")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    sub.add_parser("rollback")
    pr = sub.add_parser("promote")
    pr.add_argument("--version", required=True)
    args = ap.parse_args()

    c, n = get_client(), model_name()
    {"status": lambda: status(c, n),
     "rollback": lambda: rollback(c, n),
     "promote": lambda: promote(c, n, args.version)}[args.cmd]()
