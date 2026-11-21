"""Train on the REFERENCE period, log everything to MLflow, register the model.

Each run records: params, metrics, git commit, data hash, and the model (+ signature).
"""
import hashlib
import json
import logging
import subprocess

import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from src import registry
from src.config import DEMAND_PATH, PROCESSED_DIR, ROOT, load_params
from src.features import build_features, feature_columns

log = logging.getLogger(__name__)


def wape(y_true, y_pred) -> float:
    """Weighted absolute percentage error: sum|err| / sum(actual). Safe with zeros (unlike MAPE)."""
    y_true = np.asarray(y_true, dtype=float)
    return float(np.abs(y_true - y_pred).sum() / max(y_true.sum(), 1.0))


def evaluate(y_true, y_pred) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "wape": wape(y_true, y_pred),
    }


def make_model(tp: dict) -> Pipeline:
    """zone_id -> ordinal category (unknown zone -> NaN, handled natively), rest passthrough."""
    prep = ColumnTransformer(
        [("zone", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan), ["zone_id"])],
        remainder="passthrough",
    )
    gbm = HistGradientBoostingRegressor(
        loss=tp["loss"],
        max_iter=tp["max_iter"],
        learning_rate=tp["learning_rate"],
        max_depth=tp["max_depth"],
        categorical_features=[0],  # zone_id is first after the ColumnTransformer
        random_state=tp["random_state"],
    )
    return Pipeline([("prep", prep), ("model", gbm)])


def time_split(ref: pd.DataFrame, val_days: int):
    """Chronological split inside the reference period. Never random."""
    cutoff = ref["pickup_hour"].max() - pd.Timedelta(days=val_days)
    return ref[ref["pickup_hour"] <= cutoff], ref[ref["pickup_hour"] > cutoff]


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_info() -> dict:
    def run(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT, stderr=subprocess.DEVNULL, text=True).strip()
    try:
        return {"git_commit": run("rev-parse", "HEAD"), "git_dirty": str(bool(run("status", "--porcelain")))}
    except (OSError, subprocess.CalledProcessError):  # no git binary, or not a git checkout
        return {"git_commit": "unknown", "git_dirty": "unknown"}


def flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        out.update(flatten(v, key + ".") if isinstance(v, dict) else {key: v})
    return out


def main() -> None:
    params = load_params()
    lags, tp, mp = params["features"]["lags"], params["train"], params["mlflow"]
    cols = feature_columns(lags)

    df = pd.read_parquet(DEMAND_PATH)
    feats = build_features(df, lags)
    ref = feats[feats["period"] == "reference"]
    train, val = time_split(ref, tp["val_days"])
    log.info("train=%d rows, val=%d rows", len(train), len(val))

    model = make_model(tp).fit(train[cols], train["target"])
    pred = model.predict(val[cols])

    metrics = {f"val_{k}": v for k, v in evaluate(val["target"], pred).items()}
    # baseline = "same hour last week" (must be beaten, or the model is useless)
    base = {f"baseline_val_{k}": v for k, v in evaluate(val["target"], val["lag_168"]).items()}
    metrics.update(base)

    client = registry.get_client()  # sets the tracking URI
    mlflow.set_experiment(mp["experiment"])
    with mlflow.start_run() as run:
        mlflow.set_tags({**git_info(), "data_sha256": file_sha256(DEMAND_PATH), "training_period": "reference"})
        mlflow.log_params(flatten(params))
        mlflow.log_params({"n_train": len(train), "n_val": len(val)})
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(ROOT / "params.yaml"))
        mlflow.log_artifact(str(PROCESSED_DIR / "quality_report.json"))

        example = val[cols].head(5)
        mlflow.sklearn.log_model(
            model,
            name="model",
            signature=infer_signature(example, model.predict(example)),
            input_example=example,
            registered_model_name=mp["registered_model"],
            serialization_format="cloudpickle",  # our own model; skops rejects HistGradientBoosting trees
        )

    # find the version we just registered, tag it as candidate, auto-promote if none exists
    versions = client.search_model_versions(f"name='{mp['registered_model']}' and run_id='{run.info.run_id}'")
    version = versions[0].version
    client.set_registered_model_alias(mp["registered_model"], registry.CANDIDATE, version)
    if registry.champion_version(client, mp["registered_model"]) is None:
        registry.promote(client, mp["registered_model"], version)

    # newline="\n": the same bytes on Windows and macOS, so DVC sees the same hash everywhere
    (ROOT / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", newline="\n")
    log.info("run=%s version=%s metrics=%s", run.info.run_id, version, json.dumps(metrics))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
