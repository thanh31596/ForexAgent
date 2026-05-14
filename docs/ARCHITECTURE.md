# Architecture

This document describes the runtime architecture of ForexAgent — how data
flows through the system, what each component is responsible for, and how the
pieces are deployed. Design *rationale* (why these choices were made over
alternatives) lives in [`design_decisions.md`](design_decisions.md).

---

## 1. Overview

ForexAgent is a two-stage prediction system. A classical ML pipeline produces
a numeric forecast (log-return next bar); an agentic reasoning layer consumes
that forecast, retrieves recent market news as context, and emits a discrete
trading signal with a confidence score and a natural-language rationale.

Both stages are exposed behind a single FastAPI service. Predictions, agent
decisions, and latency measurements are written to a JSONL metrics sink and
consumed by an evaluation framework that compares the system against three
baselines and detects drift.

---

## 2. Data flow

```
  Market data            Feature                    Model                   Agent
  fetchers      ─────►   engineering    ─────►     prediction    ─────►   reasoning   ─────►   Trading
  (Alpha                 (RSI, MACD,                (LightGBM /              + RAG               signal
   Vantage,               Bollinger,                 LSTM via                (news               (BUY /
   yfinance)              MA, ATR,                   MLflow                   context)           SELL /
                          volatility)                registry)                                    HOLD,
                                                                                                  conf.,
                                                                                                  reason)
       │                       │                         │                       │                  │
       ▼                       ▼                         ▼                       ▼                  ▼
  ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
  │                          JSON logs  +  latency metrics (JSONL sink)                              │
  └──────────────────────────────────────────────────────────────────────────────────────────────────┘
                                                  │
                                                  ▼
                                          Evaluation framework
                                          (baselines, rolling
                                           metrics, drift)
```

The pipeline is fundamentally batch-oriented: data is fetched periodically,
features computed, predictions cached, signals emitted on request. The API is
a thin facade over the cached prediction plus the on-demand agent step.

---

## 3. Component responsibilities

### 3.1 Data layer (`src/data/`)

- **`fetcher.py`** exposes an abstract `MarketDataFetcher` with two concrete
  backends. Alpha Vantage is the default for backfill (rate-limited but
  documented); yfinance is the fallback for live updates (undocumented but
  free of quota). Retries with exponential backoff via `tenacity`.
- **`features.py`** is a pure-function library of technical indicators. No
  state, no I/O. Indicator windows are imported from `config.py`; nothing is
  inlined. Walk-forward splitting also lives here.

### 3.2 Models layer (`src/models/`)

- **`BaseModel`** (in `trainer.py`) defines the minimal interface every model
  must satisfy: `fit`, `predict`, `save`, `load`, `feature_importance`.
- **`lightgbm_model.py`** and **`lstm_model.py`** implement that interface.
  Adding a new model means adding one file plus one registry entry.
- **`trainer.py`** orchestrates walk-forward CV across all registered models,
  logs every run to MLflow, compares them on directional accuracy + RMSE +
  Sharpe, and promotes the best one to the `Production` stage of the model
  registry.

### 3.3 Agent layer (`src/agent/`)

- **`schemas.py`** declares Pydantic v2 models for every boundary: the
  prediction input from the ML layer, retrieved news chunks, intermediate
  reasoning state, and the final `TradingSignalResponse`. Validation
  failures surface as `422` at the API edge.
- **`rag.py`** retrieves recent FX news from Marketaux/Finnhub, chunks,
  embeds (OpenAI `text-embedding-3-small`), and indexes in Chroma.
- **`forex_agent.py`** is a LangGraph state machine with explicit nodes:
  `interpret_prediction → retrieve_context → reconcile → emit_signal`.
  Each node is a typed function; transitions and errors are explicit.

### 3.4 Evaluation layer (`src/evaluation/`)

- **`evaluator.py`** computes rolling-window metrics and compares against
  `buy_and_hold`, `ma_crossover`, and `random` baselines. Trading costs are
  modelled at `EVAL_TRADING_COST_BPS` basis points per round-trip; Sharpe
  uses the annual risk-free rate from config.
- **`drift.py`** computes PSI between recent and reference feature
  distributions and flags drift when PSI exceeds the configured threshold
  or rolling Sharpe degrades beyond `DRIFT_SHARPE_DEGRADATION_PCT`.

### 3.5 API layer (`src/api/`)

- **`main.py`** builds the FastAPI app, registers middleware (request-ID
  injection, latency timing, exception handling), and wires the lifespan
  (MLflow client warm-up, logging configuration).
- **`routes.py`** defines four endpoints. `POST /predict` is the hot path;
  `GET /health` is a liveness probe; `GET /metrics` reads the JSONL sink;
  `POST /retrain` triggers a background retraining task.

### 3.6 Observability layer (`src/observability/`)

- **`logger.py`** configures `python-json-logger` and returns
  `get_logger(__name__)` instances. Every log line is structured JSON with
  `timestamp`, `level`, `module`, `funcName`, plus arbitrary `extra={...}`
  context.
- **`metrics.py`** exposes a `latency_block(name)` context manager that
  records duration to the JSONL sink under a named operation. ML inference
  and agent reasoning are tracked separately so their latencies can be
  attributed independently.

---

## 4. Deployment topology

### 4.1 Local development

A single venv. SQLite for storage. MLflow file backend at `./mlruns/`. The
agent calls OpenAI directly. Run `make serve` for live reload.

### 4.2 Docker Compose

Four services on a private bridge network:

| Service     | Image                  | Purpose                          |
|-------------|------------------------|----------------------------------|
| `api`       | `forexagent:latest`    | FastAPI prediction service       |
| `postgres`  | `postgres:16-alpine`   | MLflow backend store + app DB    |
| `mlflow`    | `ghcr.io/mlflow/...`   | Tracking server + model registry |
| `minio`     | `minio/minio:latest`   | S3-compatible artifact store     |

MLflow points at Postgres for metadata and minio for artifacts. The API
service writes JSONL logs and metrics to a host-mounted volume so they
survive container restarts.

### 4.3 Production (illustrative)

The compose stack is the reference deployment. For a real production roll
the same containers would land on a managed orchestrator: GKE / EKS / Azure
Container Apps. MLflow and Postgres become managed services; minio becomes
S3 / GCS / Azure Blob.

---

## 5. Failure modes

A short non-exhaustive catalogue, with where each is handled:

- **Market data feed rate-limited or down** — fetcher retries with backoff
  (`tenacity`), then fails over from Alpha Vantage to yfinance, then
  serves the most recent cached bars from the DB with a stale flag.
- **News API rate-limited** — RAG layer serves cached embeddings; agent
  emits the signal with `news_freshness=stale` in the response.
- **OpenAI API timeout** — agent falls back to a deterministic rule
  (`SIGNAL_BUY_THRESHOLD` / `SIGNAL_SELL_THRESHOLD` on the raw model
  probability) and flags `agent_status=fallback`.
- **MLflow registry empty** — `/predict` returns `503`; `/retrain` is
  called manually or by CI to populate the registry.
- **Drift detected** — `/metrics` exposes a `drift_detected` flag;
  alerting integrations would page in a real deployment. Retraining is
  not automatic — drift detection is informational, not actuating.
