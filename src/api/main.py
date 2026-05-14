"""FastAPI application entrypoint."""

from __future__ import annotations

import argparse
import json
import time
import uuid
from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import Any

from fastapi import FastAPI, Request

from src.api.routes import router, run_prediction
from src.config import (
    API_DESCRIPTION,
    API_TITLE,
    API_VERSION,
    CurrencyPair,
    DataInterval,
)
from src.data.db import init_db
from src.observability.logger import configure_logging, get_logger
from src.observability.metrics import append_metric

_log = get_logger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None, None]:
    configure_logging()
    init_db()
    _log.info("api_startup")
    yield
    _log.info("api_shutdown")


app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=API_DESCRIPTION,
    lifespan=lifespan,
)
app.include_router(router)


@app.middleware("http")
async def add_latency_header(
    request: Request, call_next: Callable[[Request], Awaitable[Any]]
) -> Any:
    rid = str(uuid.uuid4())
    request.state.request_id = rid
    t0 = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception as exc:  # noqa: BLE001
        append_metric(
            {
                "event": "http_error",
                "path": str(request.url.path),
                "request_id": rid,
                "duration_ms": round((time.perf_counter() - t0) * 1000, 3),
                "error": repr(exc),
            }
        )
        raise
    dt = round((time.perf_counter() - t0) * 1000, 3)
    response.headers["X-Request-ID"] = rid
    response.headers["X-Process-Time-Ms"] = str(dt)
    append_metric(
        {
            "event": "http",
            "path": str(request.url.path),
            "method": request.method,
            "request_id": rid,
            "duration_ms": dt,
            "status_code": response.status_code,
        }
    )
    return response


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predict-sample", action="store_true")
    args = parser.parse_args()
    if args.predict_sample:
        configure_logging()
        init_db()
        out = run_prediction(CurrencyPair.EUR_USD.value, DataInterval.ONE_DAY.value)
        print(json.dumps(out, indent=2, default=str))
        return


if __name__ == "__main__":
    main()
