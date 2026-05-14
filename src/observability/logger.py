"""JSON structured logging via ``python-json-logger``."""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from typing import Any

from pythonjsonlogger import jsonlogger

from src.config import LOG_DIR, LOG_FILENAME, settings

_CONFIGURED = False


def configure_logging() -> None:
    """Configure root logger once: JSON to stdout; optional rotating file."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    root = logging.getLogger()
    root.setLevel(settings.log_level.value)
    root.handlers.clear()

    formatter = jsonlogger.JsonFormatter(  # type: ignore[attr-defined]
        "%(asctime)s %(levelname)s %(name)s %(module)s %(funcName)s %(lineno)s %(message)s",
        rename_fields={
            "levelname": "level",
            "asctime": "timestamp",
        },
    )

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    root.addHandler(sh)

    if settings.log_to_file:
        path = LOG_DIR / LOG_FILENAME
        fh = RotatingFileHandler(path, maxBytes=10_000_000, backupCount=3)
        fh.setFormatter(formatter)
        root.addHandler(fh)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a child logger emitting structured JSON lines."""
    configure_logging()
    return logging.getLogger(name)


def log_extra(**kwargs: Any) -> dict[str, Any]:
    """Helper for ``logger.info("msg", extra=log_extra(request_id=...))``."""
    return kwargs
