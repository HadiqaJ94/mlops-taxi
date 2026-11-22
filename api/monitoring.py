"""API monitoring (task T10): Prometheus metrics, live PSI gauges, JSON logs with request_id.

Four layers (session 4):
  1 infrastructure  taxi_request_seconds (latency histogram, by endpoint and status)
  2 input           taxi_input_psi{feature}: last WINDOW live rows vs the reference period
  3 prediction      taxi_prediction_psi, taxi_prediction_mean (and the reference mean to compare)
  4 outcome         not live (labels arrive late): `uv run python -m src.drift` gives WAPE per week

The PSI maths is NOT rewritten here: bin_edges / proportions / psi_from_profile come from
src/drift.py (T11), so the gauge and the drift report always agree.

The API runs ONE uvicorn worker (agreed), so the default in-process Prometheus registry is correct.
Routes are plain `def`, run in a thread pool: the rolling windows are protected by a lock.
"""
import json
import logging
import os
import threading
import time
from collections import deque
from contextvars import ContextVar

import numpy as np
import pandas as pd
from fastapi import APIRouter, Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

from src.config import DEMAND_PATH, load_params
from src.drift import bin_edges, proportions, psi_from_profile
from src.features import build_features, feature_columns

log = logging.getLogger("api")
access_log = logging.getLogger("api.access")

# ---------- metrics ----------

# Layer 1: infrastructure. `endpoint` is the route template (never the raw URL): low cardinality.
REQUEST_SECONDS = Histogram(
    "taxi_request_seconds", "Request latency in seconds", ["endpoint", "status"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6],  # 0.2 = the latency budget
)
PREDICTIONS = Counter("taxi_predictions_total", "Predictions served (one per zone-hour)", ["model_version"])
MODEL_LOADED = Gauge("taxi_model_loaded", "1 when a champion model is loaded, 0 otherwise")
# Layer 2: input. `feature` is a label (a handful of values). A request_id must NEVER be a label.
INPUT_PSI = Gauge("taxi_input_psi", "PSI of the last window of live inputs vs the reference period",
                  ["feature"])
# Layer 3: prediction
PREDICTION_PSI = Gauge("taxi_prediction_psi", "PSI of the last window of predictions vs the reference")
PREDICTION_MEAN = Gauge("taxi_prediction_mean", "Mean predicted pickups over the last window")
REFERENCE_PREDICTION_MEAN = Gauge("taxi_reference_prediction_mean",
                                  "Mean predicted pickups of the loaded model on the reference period")

MODEL_LOADED.set(0)  # until the first successful load

# An unlabeled Gauge that was never set is exported as 0.0, which Prometheus reads as a real value:
# a mean prediction of 0 would fire PredictionMeanShift at startup. NaN means "no data yet":
# every comparison with NaN is false, so no alert, and Grafana draws nothing.
NAN = float("nan")
for _gauge in (PREDICTION_PSI, PREDICTION_MEAN, REFERENCE_PREDICTION_MEAN):
    _gauge.set(NAN)


# ---------- live monitor ----------

class Monitor:
    """Rolling window of live inputs and predictions, compared with the reference period."""

    def __init__(self, window: int | None = None, every: int | None = None):
        self.window = window or int(os.getenv("MONITOR_WINDOW", "300"))       # rows kept
        self.every = every or int(os.getenv("MONITOR_PSI_EVERY", "20"))       # recompute every N rows
        self.lock = threading.Lock()
        self.features: pd.DataFrame | None = None   # the whole feature table (all periods)
        self.names: list[str] = []                  # monitored features
        self.cols: list[str] = []                   # model input columns
        self.bins = 10
        self.reference: pd.DataFrame | None = None  # rows the live traffic is compared with
        self.profiles: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.pred_profile: tuple[np.ndarray, np.ndarray] | None = None
        self.inputs: dict[str, deque] = {}
        self.preds: deque = deque(maxlen=self.window)
        self.since = 0

    # -- reference --

    def set_reference(self, reference: pd.DataFrame, names: list[str], cols: list[str], bins: int = 10):
        """Profile (decile edges + shares) of each monitored feature, computed ONCE per reference.

        Everything derived from the previous reference is dropped: the live windows are emptied and
        the gauges go back to NaN, so an alert on the old reference clears instead of going stale.
        """
        profiles = {}
        for name in names:
            edges = bin_edges(reference[name], bins)
            profiles[name] = (edges, proportions(reference[name], edges))
        with self.lock:
            self.reference, self.cols, self.bins, self.profiles = reference, cols, bins, profiles
            self.inputs = {name: deque(maxlen=self.window) for name in names}
            self.preds = deque(maxlen=self.window)
            self.pred_profile = None
            self.since = 0
        for name in names:
            INPUT_PSI.labels(name).set(NAN)
        PREDICTION_PSI.set(NAN)
        PREDICTION_MEAN.set(NAN)

    def set_features(self, features: pd.DataFrame, names: list[str], cols: list[str], bins: int = 10):
        """Keep the whole feature table; the reference is picked from it by use_window()."""
        self.features, self.names, self.cols, self.bins = features, names, cols, bins
        self.use_window(None)

    def use_window(self, window) -> str:
        """Reference = the model's training window (start, end), both included.

        None -> the reference period of the project (the first model). An empty or unknown window
        falls back to the reference period rather than monitoring against nothing.
        """
        feats = self.features
        rows = feats[feats["period"] == "reference"]
        label = "reference period"
        if window is not None:
            start, end = window
            picked = feats[(feats["pickup_hour"] >= start) & (feats["pickup_hour"] <= end)]
            if picked.empty:
                log.warning("training window %s..%s not in the data: using the reference period", start, end)
            else:
                rows, label = picked, f"training window {start}..{end}"
        self.set_reference(rows[self.cols], self.names, self.cols, self.bins)
        return f"{label}, {len(rows)} rows"

    def load_reference(self) -> None:
        """Feature table from demand.parquet, same features as src/drift.py.

        The API still starts when the file is missing (CI, a bare container): no PSI then.
        """
        if self.features is not None:
            return
        if not DEMAND_PATH.exists():
            log.warning("no %s: input and prediction PSI gauges are disabled", DEMAND_PATH)
            return
        params = load_params()
        lags, cfg = params["features"]["lags"], params["drift"]
        cols = feature_columns(lags)
        names = [n for n in cfg["features"] if n in cols]  # only what the API sees at serving time
        feats = build_features(pd.read_parquet(DEMAND_PATH), lags)
        self.set_features(feats[[*cols, "pickup_hour", "period"]], names, cols, cfg["bins"])
        log.info("PSI features loaded: %d rows, monitored %s", len(feats), names)

    def on_model_loaded(self, model, version: str, train_window=None) -> None:
        """Called at startup and after every /admin/reload.

        The reference follows the model: the first model was trained on the reference period, a
        retrained one on a recent window (MLflow tags train_start / train_end, see train_window()).
        Without this, a retrained model would be compared with Jan-Feb forever and the alerts
        would never clear.
        """
        MODEL_LOADED.set(1)
        if self.features is None:
            return
        what = self.use_window(train_window)
        try:
            ref_pred = np.clip(model.predict(self.reference[self.cols]), 0, None)
        except Exception:
            log.exception("cannot score the reference: prediction PSI disabled")
            return
        edges = bin_edges(ref_pred, self.bins)
        with self.lock:
            self.pred_profile = (edges, proportions(ref_pred, edges))
        REFERENCE_PREDICTION_MEAN.set(float(ref_pred.mean()))
        log.info("model v%s loaded, reference = %s, reference prediction mean %.2f",
                 version, what, ref_pred.mean())

    # -- live --

    def observe(self, features: pd.DataFrame, predictions, version: str) -> None:
        """Called once per scored request (single or batch). `features` has one row per prediction."""
        n = len(predictions)
        PREDICTIONS.labels(version).inc(n)
        with self.lock:
            self.preds.extend(float(p) for p in predictions)
            for name, window in self.inputs.items():
                window.extend(features[name].to_numpy(dtype=float))
            self.since += n
            if len(self.preds) < self.window or self.since < self.every:
                return  # window not full yet, or too soon: the gauges stay NaN, nothing is faked
            self.since = 0
            for name, (edges, props) in self.profiles.items():
                INPUT_PSI.labels(name).set(psi_from_profile(edges, props, np.array(self.inputs[name])))
            current = np.array(self.preds)
            PREDICTION_MEAN.set(float(current.mean()))
            if self.pred_profile is not None:
                PREDICTION_PSI.set(psi_from_profile(*self.pred_profile, current))


def train_window(client, model_name: str, version: str):
    """(start, end) of the training window of a model version, from the tags of its MLflow run.

    train.py / retrain.py must log the tags `train_start` and `train_end` (ISO hours, first and last
    pickup_hour of the training rows). Returns None when they are missing (models trained before
    this contract): the monitor then uses the reference period.
    """
    try:
        run_id = client.get_model_version(model_name, str(version)).run_id
        tags = client.get_run(run_id).data.tags
        if "train_start" in tags and "train_end" in tags:
            return pd.Timestamp(tags["train_start"]), pd.Timestamp(tags["train_end"])
        log.info("v%s has no train_start/train_end tags: reference = reference period", version)
    except Exception as exc:  # noqa: BLE001  (MLflow down or run deleted: fall back, never crash)
        log.warning("cannot read the training window of v%s: %s", version, exc)
    return None


monitor = Monitor()


# ---------- request metrics + JSON logs ----------

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")
_QUIET = {"/metrics", "/health"}  # scraped every few seconds: no access-log line


def record_request(request: Request, status_code: int, seconds: float) -> None:
    """Layer 1. Call from the HTTP middleware once the response exists."""
    route = request.scope.get("route")
    endpoint = route.path if route is not None else "other"  # 404 scans must not create new series
    if endpoint == "/metrics":
        return
    REQUEST_SECONDS.labels(endpoint, str(status_code)).observe(seconds)
    if endpoint not in _QUIET:
        access_log.info("request", extra={"fields": {
            "method": request.method, "path": endpoint, "status": status_code,
            "ms": round(seconds * 1000, 1)}})


class JsonFormatter(logging.Formatter):
    """One JSON object per line, always carrying the request_id of the request being served."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname, "logger": record.name, "msg": record.getMessage(),
            "request_id": request_id_var.get(),
        }
        payload.update(getattr(record, "fields", {}))
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def setup_logging() -> None:
    logger = logging.getLogger("api")
    if any(isinstance(h.formatter, JsonFormatter) for h in logger.handlers):
        return  # already configured (the lifespan runs once per TestClient)
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logger.handlers = [handler]
    logger.setLevel(logging.INFO)
    logger.propagate = False


router = APIRouter()


@router.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
