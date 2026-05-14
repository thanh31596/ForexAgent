"""HTTP routes for predict / health / metrics / retrain."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

from src.agent.forex_agent import run_agent
from src.agent.schemas import PredictionPayload, TradingSignalResponse
from src.config import (
    DEFAULT_BACKFILL_DAYS,
    DataInterval,
    settings,
)
from src.data.db import init_db
from src.data.features import add_features, feature_columns
from src.data.fetcher import fetch_and_persist
from src.evaluation.drift import drift_flags
from src.models.base import ClassicalModel
from src.models.lightgbm_model import LightGBMForexModel
from src.models.lstm_model import LSTMForexModel
from src.models.trainer import BEST_MODEL_META
from src.observability.logger import get_logger
from src.observability.metrics import read_latency_tail

_log = get_logger(__name__)
router = APIRouter()


class PredictRequest(BaseModel):
    pair: str = Field(examples=["EURUSD=X"])
    interval: str = Field(default=DataInterval.ONE_DAY.value)


def _load_serving_model() -> tuple[ClassicalModel | None, dict[str, Any] | None]:
    if not BEST_MODEL_META.exists():
        return None, None
    meta = json.loads(BEST_MODEL_META.read_text(encoding="utf-8"))
    path = Path(meta["path"])
    name = meta["model"]
    if name == "lightgbm":
        return LightGBMForexModel.load(path), meta
    if name == "lstm":
        return LSTMForexModel.load(path), meta
    return None, None


def _sigmoid_p_up(pred_log_ret: float) -> float:
    return float(1.0 / (1.0 + math.exp(-pred_log_ret * 25.0)))


def run_prediction(pair: str, interval: str) -> dict[str, Any]:
    model, meta = _load_serving_model()
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="No trained model artifact found. Run `make train` first.",
        )
    raw = fetch_and_persist(pair, interval, days=DEFAULT_BACKFILL_DAYS)
    if raw.empty:
        raise HTTPException(status_code=503, detail="No market data available.")
    feats = add_features(raw).dropna()
    cols = feature_columns(feats)
    X = feats[cols]
    yhat = model.predict(X)
    last = float(yhat[-1]) if np.isfinite(yhat[-1]) else float(np.nanmean(yhat))
    if math.isnan(last):
        raise HTTPException(status_code=503, detail="Model returned no valid prediction.")
    p_up = _sigmoid_p_up(last)
    payload = PredictionPayload(
        pair=pair,
        predicted_log_return=last,
        p_up=p_up,
    )
    sig: TradingSignalResponse = run_agent(payload)
    return {
        "model_meta": meta,
        "prediction": payload.model_dump(),
        "signal": sig.model_dump(),
    }


@router.get("/health")
def health() -> dict[str, str]:
    init_db()
    return {"status": "ok", "env": settings.env}


@router.get("/metrics")
def metrics() -> dict[str, Any]:
    init_db()
    latency = read_latency_tail(200)
    drift: dict[str, Any] = {"drift_detected": False}
    try:
        raw = fetch_and_persist("EURUSD=X", DataInterval.ONE_DAY.value, days=600)
        feats = add_features(raw).dropna()
        drift = drift_flags(feats)
    except Exception as exc:  # noqa: BLE001
        drift = {"drift_detected": False, "error": repr(exc)}
    return {"latency": latency, "drift": drift}


@router.post("/predict")
def predict(body: PredictRequest) -> dict[str, Any]:
    return run_prediction(body.pair, body.interval)


@router.post("/retrain")
def retrain(background_tasks: BackgroundTasks) -> dict[str, str]:
    def job() -> None:
        subprocess.run(
            [sys.executable, "-m", "src.models.trainer"],
            check=False,
        )

    background_tasks.add_task(job)
    return {"status": "scheduled"}


@router.get("/openapi-version")
def openapi_version() -> dict[str, str]:
    return {"openapi": "3.1"}
