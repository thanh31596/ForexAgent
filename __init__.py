"""ForexAgent: forex trading-signal system combining classical ML and agentic AI.

Submodules
----------
config         All runtime configuration and domain constants.
data           Market-data ingestion and feature engineering.
models         Classical ML models (LightGBM, LSTM) and training pipeline.
agent          LangGraph reasoning agent and RAG over market news.
api            FastAPI service exposing predict / health / metrics / retrain.
evaluation     Rolling evaluation, baselines, drift detection.
observability  Structured logging, latency tracking, JSONL metrics sink.

Usage
-----
>>> from src.config import settings
>>> settings.env
'dev'
"""

__version__ = "0.1.0"
__author__ = "Stephen Kim"
