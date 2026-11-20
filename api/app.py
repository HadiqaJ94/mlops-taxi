"""FastAPI service for the NYC taxi-demand model (task T5).

Loads the champion model once at startup from the MLflow registry and serves it
over HTTP. Features are built with src.features.serving_features - the exact same
code training uses - so there is no train/serve skew.

Run locally (MLflow must be up):
    uv run uvicorn api.app:app --reload
    # interactive docs at http://localhost:8000/docs
"""
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime

import mlflow
from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator

from src import registry
from src.config import load_params
from src.features import serving_features

_params = load_params()
LAGS = _params["features"]["lags"]
MIN_HISTORY = max(LAGS)  # longest lag (168h = 1 week) -> minimum history needed

# config from env vars, with sensible defaults (never hard-code the model name)
MODEL_NAME = os.getenv("MODEL_NAME", _params["mlflow"]["registered_model"])
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "champion")

# the loaded model and its version live here, filled at startup
state: dict = {}


def load_model() -> None:
    """(Re)load the aliased model and remember its version.

    Called once at startup and again by POST /admin/reload (for rollback, T18).
    """
    client = registry.get_client()  # also sets the MLflow tracking URI from .env
    state["model_version"] = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS).version
    state["model"] = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@{MODEL_ALIAS}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()  # load once, before the service accepts traffic
    yield


app = FastAPI(title="NYC taxi demand", version="1.0.0", lifespan=lifespan)


# ---- request / response schemas (Pydantic validates every request) ----

class PredictRequest(BaseModel):
    zone_id: int = Field(ge=1, le=263, description="NYC TLC pickup zone id")
    target_hour: datetime = Field(description="the hour to forecast, ISO 8601")
    recent_counts: list[float] = Field(
        min_length=MIN_HISTORY,
        description=f"hourly pickup counts, oldest->newest; at least {MIN_HISTORY} values",
    )

    @field_validator("recent_counts")
    @classmethod
    def counts_non_negative(cls, v: list[float]) -> list[float]:
        if any(c < 0 for c in v):
            raise ValueError("recent_counts must all be >= 0")
        return v


class Prediction(BaseModel):
    predicted_pickups: float
    model_version: str
    model_ms: float  # time spent inside the model, milliseconds


class BatchRequest(BaseModel):
    items: list[PredictRequest]


class BatchResponse(BaseModel):
    predictions: list[Prediction]
    model_version: str


# ---- scoring helper, shared by /predict and /predict_batch ----

def _score(req: PredictRequest) -> Prediction:
    feats = serving_features(req.zone_id, req.target_hour, req.recent_counts, LAGS)
    t0 = time.perf_counter()
    pred = state["model"].predict(feats)
    model_ms = (time.perf_counter() - t0) * 1000
    return Prediction(
        predicted_pickups=float(pred[0]),
        model_version=state["model_version"],
        model_ms=round(model_ms, 2),
    )


# ---- endpoints ----

@app.post("/predict", response_model=Prediction)
def predict(req: PredictRequest) -> Prediction:
    """Forecast next-hour pickups for one zone."""
    return _score(req)


@app.post("/predict_batch", response_model=BatchResponse)
def predict_batch(req: BatchRequest) -> BatchResponse:
    """Forecast for many zones in one call. Same answers as calling /predict once each."""
    predictions = [_score(item) for item in req.items]
    return BatchResponse(predictions=predictions, model_version=state["model_version"])


@app.get("/health")
def health() -> dict:
    """Liveness + which model version is loaded. Used by CI and load balancers."""
    return {"status": "ok", "model_version": state.get("model_version")}


@app.post("/admin/reload")
def admin_reload() -> dict:
    """Reload the champion from the registry without restarting (used for rollback, T18)."""
    load_model()
    return {"status": "reloaded", "model_version": state["model_version"]}
