"""Pydantic contracts for the agent + API edges."""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.config import TradingSignal


class PredictionPayload(BaseModel):
    """Numeric forecast handed from the ML layer to the agent."""

    pair: str
    predicted_log_return: float
    p_up: float = Field(ge=0.0, le=1.0, description="Model-estimated probability of upward move")


class NewsChunk(BaseModel):
    headline: str
    source: str = "unknown"
    published_at: str | None = None


class TradingSignalResponse(BaseModel):
    pair: str
    signal: TradingSignal
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str
    news_freshness: str = "live"
    agent_status: str = "ok"
