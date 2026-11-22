"""Taxi demand API (task T5): pickups in a zone for the next hour, served by @champion.

Combines two teammates' serving work (Manel + Soumaya). Design:
- The model loads ONCE, in `lifespan`, from models:/<name>@<alias>. Never per request.
- No model -> the service still starts: /health answers 503, /predict* answer 503.
- Bad input -> 422 (the caller's fault), never a confident wrong answer.
- Every prediction carries model_version and model_ms; every response carries
  X-Request-ID and X-Process-Time-Ms (time in the model vs total server time).
- Unknown zones (not in the trained set) are refused with 422.
- Features come from src.features.serving_features, the same code as training (no skew).

Run (MLflow server up): uv run uvicorn api.app:app --port 8000
"""
import logging
import os
import secrets
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Annotated

import mlflow
import numpy as np
import pandas as pd
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src import registry
from src.config import load_params
from src.features import serving_features

log = logging.getLogger("api")

LAGS: list[int] = load_params()["features"]["lags"]
MIN_HISTORY = max(LAGS)        # 168 hours: the longest lag is "same hour last week"
MAX_HISTORY = 2 * MIN_HISTORY
MAX_BATCH = 1000

# Config from env vars (never hard-code the model name / alias).
MODEL_NAME = os.getenv("MODEL_NAME", registry.model_name())
MODEL_ALIAS = os.getenv("MODEL_ALIAS", registry.CHAMPION)

# Everything the routes need: {"model", "version", "zones"}, or None when no model is loaded.
# Reload swaps the whole dict in ONE assignment, so a request never mixes the model of one
# version with the version number of another, and never sees a half-loaded state.
state: dict = {"current": None}


def load_champion() -> dict:
    """Read which version the alias points to, then load exactly that version."""
    client = registry.get_client()  # sets the tracking URI from MLFLOW_TRACKING_URI
    version = str(client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS).version)
    # Load by version, not by alias: the alias could move between the two calls.
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{version}")
    # The zones the model was trained on live in its zone encoder: no data file needed.
    encoder = model.named_steps["prep"].named_transformers_["zone"]
    zones = frozenset(int(z) for z in encoder.categories_[0])
    log.info("loaded %s v%s (%d zones)", MODEL_NAME, version, len(zones))
    return {"model": model, "version": version, "zones": zones}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        state["current"] = load_champion()
    except Exception:  # start degraded rather than not at all: /health says why
        log.exception("no model loaded, the service starts degraded")
    yield
    state["current"] = None


app = FastAPI(
    title="NYC taxi demand API",
    version="1.0.0",
    description="Pickups per zone for the next hour. Data: NYC TLC Trip Record Data.",
    lifespan=lifespan,
)


@app.middleware("http")
async def request_context(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
    start = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-Ms"] = f"{(time.perf_counter() - start) * 1000:.1f}"
    return response


# ---------- contract ----------

class DemandRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")  # a typo in a field name is a 422, not ignored

    zone_id: int = Field(ge=1, le=263, description="TLC pickup zone (LocationID).", examples=[237])
    target_hour: datetime = Field(
        description="Start of the hour to forecast, New York local time, on the hour.",
        examples=["2020-02-20T18:00:00"],
    )
    recent_counts: list[Annotated[float, Field(ge=0)]] = Field(
        min_length=MIN_HISTORY, max_length=MAX_HISTORY,
        description=(f"Pickups per hour in this zone, oldest first; the last value is the hour "
                     f"before target_hour. At least {MIN_HISTORY} values (one week)."),
        examples=[[30.0] * MIN_HISTORY],
    )

    @field_validator("target_hour")
    @classmethod
    def on_the_hour(cls, v: datetime) -> datetime:
        if v.tzinfo is not None:  # the model works in New York local time
            v = pd.Timestamp(v).tz_convert("America/New_York").tz_localize(None).to_pydatetime()
        if (v.minute, v.second, v.microsecond) != (0, 0, 0):
            raise ValueError("target_hour must be on the hour, e.g. 2020-02-20T18:00:00")
        return v


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    items: list[DemandRequest] = Field(min_length=1, max_length=MAX_BATCH)


class Prediction(BaseModel):
    zone_id: int
    target_hour: datetime
    predicted_pickups: float = Field(description="Expected pickups in target_hour, >= 0.")
    model_version: str
    model_ms: float  # time spent inside model.predict, milliseconds


class BatchResponse(BaseModel):
    model_version: str
    predictions: list[Prediction]


# ---------- prediction ----------

def predict_many(items: list[DemandRequest]) -> tuple[str, list[Prediction]]:
    current = state["current"]  # one snapshot, even if /admin/reload runs meanwhile
    if current is None:
        raise HTTPException(status_code=503, detail="model not loaded")

    unknown = sorted({it.zone_id for it in items} - current["zones"])
    if unknown:
        raise HTTPException(
            status_code=422,
            detail=f"zone_id {unknown} not served: the model knows {len(current['zones'])} zones",
        )

    features = pd.concat(
        [serving_features(it.zone_id, it.target_hour, it.recent_counts, LAGS) for it in items],
        ignore_index=True,
    )
    t0 = time.perf_counter()
    y = np.clip(current["model"].predict(features), 0, None)  # counts cannot be negative
    model_ms = round((time.perf_counter() - t0) * 1000, 2)
    version = current["version"]
    return version, [
        Prediction(zone_id=it.zone_id, target_hour=it.target_hour,
                   predicted_pickups=round(float(p), 3), model_version=version, model_ms=model_ms)
        for it, p in zip(items, y)
    ]


# Plain `def` routes run in a thread pool, so model.predict and the MLflow download in
# /admin/reload do not block the event loop.

@app.post("/predict", response_model=Prediction)
def predict(req: DemandRequest) -> Prediction:
    return predict_many([req])[1][0]


@app.post("/predict_batch", response_model=BatchResponse)
def predict_batch(req: BatchRequest) -> BatchResponse:
    version, predictions = predict_many(req.items)
    return BatchResponse(model_version=version, predictions=predictions)


@app.get("/health")
def health():
    """200 when a model is loaded, 503 otherwise: a load balancer stops sending traffic."""
    current = state["current"]
    if current is None:
        return JSONResponse(status_code=503, content={"status": "degraded", "model_version": None})
    return {"status": "ok", "model_version": current["version"]}


DEFAULT_ADMIN_TOKEN = "change-me"  # value shipped in .env.example: never accepted


@app.post("/admin/reload")
def reload_model(x_admin_token: str | None = Header(default=None)):
    """Re-read the alias and swap the model in without a restart (used after promote/rollback, T18).

    Requires header X-Admin-Token == ADMIN_TOKEN (from .env). On failure the current model keeps serving.
    """
    expected = os.getenv("ADMIN_TOKEN")
    if not expected or expected == DEFAULT_ADMIN_TOKEN:  # the .env.example value is public
        raise HTTPException(status_code=503,
                            detail="admin routes disabled: set your own ADMIN_TOKEN in .env (not change-me)")
    if not x_admin_token or not secrets.compare_digest(x_admin_token, expected):
        raise HTTPException(status_code=401, detail="invalid or missing X-Admin-Token")

    previous = state["current"]["version"] if state["current"] else None
    try:
        new = load_champion()
    except Exception as exc:
        log.exception("reload failed, keeping v%s", previous)
        raise HTTPException(status_code=503, detail=f"reload failed, still serving v{previous}: {exc}")
    state["current"] = new
    return {"previous_version": previous, "model_version": new["version"]}
