"""Train a model, log everything to MLflow, register it as @candidate.

Training never touches @champion: promotion is the training gate's decision (src/gate.py, T6).
Each run records: params, metrics, git commit, data hash, and the model (+ signature).

Reused by retraining (src/retrain.py, T16): window, train_model, score, log_and_register.
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
    """Chronological split inside the reference period: the last `val_days` days are validation.

    src/drift.py uses the same rule; params.yaml `gate` holds the same window as fixed dates.
    """
    cutoff = ref["pickup_hour"].max() - pd.Timedelta(days=val_days)
    return ref[ref["pickup_hour"] <= cutoff], ref[ref["pickup_hour"] > cutoff]


def window(feats: pd.DataFrame, start, end) -> pd.DataFrame:
    """Rows with start <= pickup_hour < end (end excluded, so consecutive windows never overlap)."""
    hours = feats["pickup_hour"]
    return feats[(hours >= pd.Timestamp(start)) & (hours < pd.Timestamp(end))]


def train_model(feats: pd.DataFrame, train_start, train_end, params: dict) -> Pipeline:
    """Fit a new model on the rows of `feats` in [train_start, train_end).

    feats: output of build_features on the full hourly series. A row's lags are earlier hours of
    the same zone, so with train_end <= validation start nothing from validation is used.
    """
    cols = feature_columns(params["features"]["lags"])
    train = window(feats, train_start, train_end)
    if train.empty:
        raise ValueError(f"no training rows between {train_start} and {train_end}")
    log.info("training on %s..%s (%d rows)", train_start, train_end, len(train))
    return make_model(params["train"]).fit(train[cols], train["target"])


def score(model, val: pd.DataFrame, lags) -> dict:
    """Model metrics on `val`, plus the 'same hour last week' baseline (must be beaten)."""
    pred = model.predict(val[feature_columns(lags)])
    metrics = {f"val_{k}": v for k, v in evaluate(val["target"], pred).items()}
    metrics.update({f"baseline_val_{k}": v for k, v in evaluate(val["target"], val["lag_168"]).items()})
    return metrics


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


def log_and_register(model, params: dict, metrics: dict, example: pd.DataFrame,
                     tags: dict, extra_params: dict | None = None) -> tuple[str, str]:
    """Log one MLflow run, register the model, point @candidate at it. Returns (run_id, version).

    Never moves @champion: run `uv run python -m src.gate` to decide on a promotion.
    """
    mp = params["mlflow"]
    client = registry.get_client()  # sets the tracking URI
    mlflow.set_experiment(mp["experiment"])
    with mlflow.start_run() as run:
        mlflow.set_tags({**git_info(), "data_sha256": file_sha256(DEMAND_PATH), **tags})
        mlflow.log_params(flatten(params))
        mlflow.log_params(extra_params or {})
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(ROOT / "params.yaml"))
        mlflow.log_artifact(str(PROCESSED_DIR / "quality_report.json"))
        mlflow.sklearn.log_model(
            model,
            name="model",
            signature=infer_signature(example, model.predict(example)),
            input_example=example,
            registered_model_name=mp["registered_model"],
            serialization_format="cloudpickle",  # our own model; skops rejects HistGradientBoosting trees
        )

    versions = client.search_model_versions(f"name='{mp['registered_model']}' and run_id='{run.info.run_id}'")
    version = str(versions[0].version)
    client.set_registered_model_alias(mp["registered_model"], registry.CANDIDATE, version)
    return run.info.run_id, version


def main() -> None:
    params = load_params()
    lags = params["features"]["lags"]
    feats = build_features(pd.read_parquet(DEMAND_PATH), lags)
    ref = feats[feats["period"] == "reference"]

    # validation = the gate's fixed window; training = everything in the reference period before it
    val_start, val_end = pd.Timestamp(params["gate"]["val_start"]), pd.Timestamp(params["gate"]["val_end"])
    train_start = ref["pickup_hour"].min()
    model = train_model(ref, train_start, val_start, params)
    val = window(ref, val_start, val_end)
    metrics = score(model, val, lags)
    n_train = len(window(ref, train_start, val_start))
    log.info("train=%d rows, val=%d rows", n_train, len(val))

    run_id, version = log_and_register(
        model, params, metrics, val[feature_columns(lags)].head(5),
        tags={"training_period": "reference", "trigger": "pipeline",
              "train_window": f"{train_start}..{val_start}", "val_window": f"{val_start}..{val_end}"},
        extra_params={"n_train": n_train, "n_val": len(val)},
    )

    # newline="\n": the same bytes on Windows and macOS, so DVC sees the same hash everywhere
    (ROOT / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", newline="\n")
    log.info("run=%s version=%s metrics=%s", run_id, version, json.dumps(metrics))
    log.info("v%s is @candidate. Promotion is the gate's call: uv run python -m src.gate", version)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
