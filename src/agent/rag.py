"""News retrieval + optional Chroma vector store (stub-friendly without API keys)."""

from __future__ import annotations

import hashlib
from typing import Any

import requests

from src.config import settings
from src.observability.logger import get_logger

_log = get_logger(__name__)


def _stub_headlines(pair: str) -> list[str]:
    return [
        f"[stub] Macro desk notes steady liquidity in {pair} ahead of NY session.",
        f"[stub] Volatility compressed on {pair} as majors await CPI calendar risk.",
    ]


def fetch_marketaux_headlines(pair: str) -> list[str]:
    key = settings.marketaux_api_key
    if key is None or not key.get_secret_value():
        return []
    q = pair.replace("=X", "")
    url = "https://api.marketaux.com/v1/news/all"
    params: dict[str, Any] = {
        "api_token": key.get_secret_value(),
        "symbols": q[:6],
        "filter_entities": "true",
        "language": "en",
    }
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    out: list[str] = []
    for item in data.get("data", [])[:8]:
        out.append(str(item.get("title", "")))
    return [h for h in out if h]


def retrieve_news(pair: str) -> tuple[list[str], str]:
    """Return ``(headlines, freshness_label)``."""
    headlines = fetch_marketaux_headlines(pair)
    if headlines:
        return headlines, "live"
    headlines = _stub_headlines(pair)
    return headlines, "stale"


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embeddings for RAG; deterministic hash fallback if OpenAI is unavailable."""
    if settings.openai_api_key and settings.openai_api_key.get_secret_value():
        try:
            from openai import OpenAI

            client = OpenAI(api_key=settings.openai_api_key.get_secret_value())
            resp = client.embeddings.create(
                model="text-embedding-3-small",
                input=texts,
            )
            return [d.embedding for d in resp.data]
        except Exception as exc:  # noqa: BLE001
            _log.warning("openai_embed_failed", extra={"error": repr(exc)})
    out: list[list[float]] = []
    for t in texts:
        h = hashlib.sha256(t.encode("utf-8")).digest()
        vec = [((b - 128) / 128.0) for b in h[:16]]
        out.append(vec)
    return out


def rag_context(pair: str) -> str:
    headlines, freshness = retrieve_news(pair)
    joined = "\n- ".join(headlines)
    return f"news_freshness={freshness}\n- {joined}"
