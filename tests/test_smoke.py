"""Tests for deterministic helpers (no external services)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from src.api.main import app
from src.data.features import add_features
from src.evaluation.drift import psi


def test_add_features_creates_indicators() -> None:
    idx = pd.date_range("2020-01-01", periods=400, freq="D")
    rng = np.linspace(1.0, 1.6, len(idx))
    df = pd.DataFrame(
        {
            "Open": rng,
            "High": rng * 1.01,
            "Low": rng * 0.99,
            "Close": rng,
            "Volume": 1e6,
        },
        index=idx,
    )
    out = add_features(df).dropna()
    assert "rsi" in out.columns
    assert "y" in out.columns


def test_psi_zero_on_identical() -> None:
    x = np.random.default_rng(0).normal(size=500)
    assert psi(x, x) < 1e-6


def test_health_endpoint() -> None:
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
