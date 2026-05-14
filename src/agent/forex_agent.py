"""LangGraph multi-step trading agent."""

from __future__ import annotations

from typing import Any, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from src.agent.rag import retrieve_news
from src.agent.schemas import PredictionPayload, TradingSignalResponse
from src.config import (
    AGENT_MAX_TOKENS,
    AGENT_TEMPERATURE,
    SIGNAL_BUY_THRESHOLD,
    SIGNAL_SELL_THRESHOLD,
    TradingSignal,
    settings,
)
from src.observability.logger import get_logger
from src.observability.metrics import latency_block

_log = get_logger(__name__)


class AgentState(TypedDict, total=False):
    payload: PredictionPayload
    news_ctx: str
    news_freshness: str
    response: TradingSignalResponse


def _rule_based(payload: PredictionPayload, news_ctx: str, freshness: str) -> TradingSignalResponse:
    p = payload.p_up
    if p >= SIGNAL_BUY_THRESHOLD:
        sig, conf = TradingSignal.BUY, float(min(1.0, (p - 0.5) * 2))
        rationale = "Model probability skews bullish; rule-based fallback engaged."
    elif p <= SIGNAL_SELL_THRESHOLD:
        sig, conf = TradingSignal.SELL, float(min(1.0, (0.5 - p) * 2))
        rationale = "Model probability skews bearish; rule-based fallback engaged."
    else:
        sig, conf = TradingSignal.HOLD, 0.35
        rationale = "Model probability near coin-flip; standing aside."
    rationale += f" News context ({freshness}): {news_ctx[:400]}"
    return TradingSignalResponse(
        pair=payload.pair,
        signal=sig,
        confidence=conf,
        rationale=rationale,
        news_freshness=freshness,
        agent_status="fallback",
    )


def node_retrieve_context(state: AgentState) -> AgentState:
    payload = state["payload"]
    headlines, freshness = retrieve_news(payload.pair)
    ctx = "\n".join(f"- {h}" for h in headlines)
    return {"news_ctx": ctx, "news_freshness": freshness}


def node_reconcile(state: AgentState) -> AgentState:
    payload = state["payload"]
    news_ctx = state.get("news_ctx", "")
    freshness = state.get("news_freshness", "unknown")
    key = settings.openai_api_key
    if key is None or not key.get_secret_value():
        return {"response": _rule_based(payload, news_ctx, freshness)}
    try:
        with latency_block("agent.openai.reconcile", extra={"pair": payload.pair}):
            llm = ChatOpenAI(
                model=settings.agent_model,
                temperature=AGENT_TEMPERATURE,
                max_tokens=AGENT_MAX_TOKENS,
                api_key=key.get_secret_value(),
                timeout=30.0,
            )
            sys = SystemMessage(
                content=(
                    "You are a disciplined FX desk analyst. Given a model forecast and headlines, "
                    "emit a discrete trading stance with calibrated confidence. "
                    "Prefer HOLD when evidence conflicts."
                )
            )
            human = HumanMessage(
                content=(
                    f"pair={payload.pair}\n"
                    f"predicted_log_return={payload.predicted_log_return:.6f}\n"
                    f"p_up={payload.p_up:.4f}\n\n"
                    f"NEWS:\n{news_ctx}\n"
                )
            )
            structured = llm.with_structured_output(TradingSignalResponse)
            resp = structured.invoke([sys, human])
            if not isinstance(resp, TradingSignalResponse):
                raise TypeError("structured output type mismatch")
            resp.news_freshness = freshness
            resp.agent_status = "ok"
            if not resp.pair:
                resp.pair = payload.pair
            return {"response": resp}
    except Exception as exc:  # noqa: BLE001
        _log.warning("agent_llm_failed", extra={"error": repr(exc)})
        return {"response": _rule_based(payload, news_ctx, freshness)}


def build_graph() -> Any:
    g = StateGraph(AgentState)
    g.add_node("retrieve_context", node_retrieve_context)
    g.add_node("reconcile", node_reconcile)
    g.set_entry_point("retrieve_context")
    g.add_edge("retrieve_context", "reconcile")
    g.add_edge("reconcile", END)
    return g.compile()


GRAPH = build_graph()


def run_agent(payload: PredictionPayload) -> TradingSignalResponse:
    with latency_block("agent.graph.invoke", extra={"pair": payload.pair}):
        out = GRAPH.invoke({"payload": payload})
    resp = out.get("response")
    if not isinstance(resp, TradingSignalResponse):
        raise RuntimeError("Agent did not produce a TradingSignalResponse")
    _log.info("agent_done", extra={"pair": resp.pair, "signal": resp.signal.value})
    return resp
