"""JSONL latency sink + ``latency_block`` context manager."""

from __future__ import annotations

import json
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import LATENCY_METRICS_FILENAME, METRICS_DIR
from src.observability.logger import get_logger

_log = get_logger(__name__)
_lock = threading.Lock()


def _metrics_path() -> Path:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    return METRICS_DIR / LATENCY_METRICS_FILENAME


def append_metric(record: dict[str, Any]) -> None:
    """Append one JSON object as a single line (thread-safe)."""
    line = json.dumps(record, default=str) + "\n"
    path = _metrics_path()
    with _lock:
        path.open("a", encoding="utf-8").write(line)


@contextmanager
def latency_block(
    name: str,
    *,
    request_id: str | None = None,
    extra: dict[str, Any] | None = None,
) -> Iterator[None]:
    """Time a block and write duration + metadata to the JSONL metrics sink."""
    rid = request_id or str(uuid.uuid4())
    t0 = time.perf_counter()
    payload: dict[str, Any] = {
        "event": "latency",
        "operation": name,
        "request_id": rid,
        "ts": datetime.now(tz=timezone.utc).isoformat(),
    }
    if extra:
        payload["extra"] = extra
    try:
        yield
        payload["status"] = "ok"
    except Exception as exc:  # noqa: BLE001 — surface as metric + re-raise
        payload["status"] = "error"
        payload["error"] = repr(exc)
        raise
    finally:
        payload["duration_ms"] = round((time.perf_counter() - t0) * 1000, 3)
        append_metric(payload)
        _log.info("latency", extra={"operation": name, **payload})


def read_latency_tail(max_lines: int = 200) -> list[dict[str, Any]]:
    """Return the last ``max_lines`` parsed JSON objects from the latency sink."""
    path = _metrics_path()
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    out: list[dict[str, Any]] = []
    for line in lines[-max_lines:]:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out
