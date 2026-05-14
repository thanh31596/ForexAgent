"""Observability primitives.

Modules
-------
logger    JSON structured logging configured via :func:`configure_logging`,
          consumed throughout the codebase via ``get_logger(__name__)``.
metrics   Latency-tracking context manager and JSONL metrics sink. Tracks
          ML inference latency and agent reasoning latency separately so
          they can be diagnosed independently.

These primitives are imported by every other layer. They are written
alongside Component 1 (data pipeline) rather than at the end of the build,
so every component emits structured telemetry from its first commit.
"""
