"""Agentic AI reasoning layer.

Modules
-------
forex_agent   LangGraph state machine that consumes the ML prediction and
              retrieved news context, reasons in multiple steps, and emits
              a validated ``TradingSignal``.
schemas       Pydantic v2 contracts for all agent inputs, intermediate
              states, and outputs. Boundary validation lives here.
rag           News retrieval (Marketaux / Finnhub) and Chroma-indexed
              embedding store for the retrieval step.
"""
