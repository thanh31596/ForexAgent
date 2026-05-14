# Design Decisions

Architectural Decision Records (ADRs) for ForexAgent. Each entry follows the
*Context / Decision / Consequences* format. Decisions are appended chronologically
and never silently rewritten — if a decision is reversed, a new ADR records the
reversal and references the original.

---

## ADR-001: LangGraph over the legacy LangChain `AgentExecutor`

**Status:** Accepted.

**Context.** The agentic layer needs multi-step reasoning that consumes both a
numeric ML prediction and retrieved news context, then emits a structured
trading signal. The legacy `AgentExecutor` pattern is a tool-using loop with
implicit state; it is hard to test, hard to instrument per-step, and is the
target of LangChain's own deprecation guidance in favour of LangGraph.

**Decision.** Use `langgraph` for the agent's control flow with
`langchain-core` for prompt templates and `langchain-openai` for the LLM
binding. Each reasoning step is an explicit node in a typed state machine:
`interpret_prediction → retrieve_context → reconcile → emit_signal`.

**Consequences.**

- More boilerplate than `AgentExecutor`; state transitions are explicit
  rather than emergent.
- Per-node latency is straightforward to track — every node call goes
  through `latency_block` in `src/observability/metrics.py`.
- Errors are explicit transitions, not exceptions bubbling out of an opaque
  loop.
- Reviewers shipping 2026-era LangChain code will see the current idiom
  rather than a deprecated pattern.

---

## ADR-002: Walk-forward CV with expanding window as default

**Status:** Accepted.

**Context.** Forex returns are non-stationary; the relationship between
indicators and forward returns shifts with the macro regime. Random train/test
splits leak future information through both temporal ordering and shared
regime statistics. Walk-forward CV is the standard remedy, but the choice
between *expanding* and *sliding* windows has consequences.

**Decision.** Default to expanding-window walk-forward with `WALK_FORWARD_N_SPLITS=5`,
`WALK_FORWARD_TEST_SIZE=200`, and `WALK_FORWARD_GAP=1`. Sliding windows are
available via `WALK_FORWARD_STRATEGY="sliding"` for ablation.

**Consequences.**

- Earlier folds train on less data than later ones; performance metrics may
  be optimistically biased for the final fold.
- Older regimes still influence the model — appropriate for FX where
  macroeconomic patterns repeat, less appropriate for pure trend-following.
- The gap parameter prevents look-ahead bias when target horizon > 1.

---

## ADR-003: LightGBM as the primary classical model

**Status:** Accepted.

**Context.** The architecture brief listed "XGBoost/LightGBM" as the
gradient-boosting choice. The two are functionally comparable on tabular
data; the question is which to make the default.

**Decision.** LightGBM as the headline classical model. XGBoost not included
in the initial scope; can be added behind the `BaseModel` interface in a
single file if benchmarks ever justify it.

**Consequences.**

- Faster training and lower memory than XGBoost on the same data.
- Native categorical-feature handling — useful once pair, session, or
  weekday indicators enter the feature set.
- One less framework dependency than supporting both at launch.
- File named `lightgbm_model.py` rather than the `xgboost_model.py`
  originally sketched in the project structure.

---

## ADR-004: MLflow with PostgreSQL backend and minio artifact store in compose

**Status:** Accepted.

**Context.** MLflow's default backend is a local SQLite file with on-disk
artifacts. That works for solo development but does not look like a
production deployment, and one of this project's purposes is to demonstrate
production-shaped MLOps.

**Decision.** Local development still uses `file://./mlruns`. The Docker
Compose stack runs an MLflow server container with PostgreSQL as the
backend store and minio (S3-compatible) as the artifact store. The API
service reads `MLFLOW_TRACKING_URI` from environment, defaulting to local
in dev and `http://mlflow:5000` in compose.

**Consequences.**

- Heavier compose stack (four services instead of two).
- Mirrors the architecture of a real MLflow deployment on managed
  Postgres + S3.
- Allows experimentation with model promotion via the registry's REST
  API rather than just the local file system.

---

## ADR-005: `src/` is itself the importable package

**Status:** Accepted.

**Context.** The conventional layout for a Python project is
`src/<package_name>/`, so that the installed package has a distinct name.
The project structure brief used `src/` as the package directly, so
imports look like `from src.config import settings`.

**Decision.** Keep the `src/` layout as authored. Configure
`hatch.build.targets.wheel` with `packages = ["src"]`. Document the
non-standard layout in the README.

**Consequences.**

- Reviewers familiar with the conventional layout may flag it.
- Imports inside the project are stable regardless of installation state.
- Migrating to `src/forexagent/` later is a single-PR refactor: move
  files, update `hatch` config, find-and-replace `src.` → `forexagent.`.

---

## ADR-006: Pydantic v2 at every external boundary

**Status:** Accepted.

**Context.** The project takes input from environment variables (config),
HTTP requests (API), and LLM outputs (agent). All three are sources of
malformed or unexpected data. Hand-rolled `if isinstance(...)` checks
become the dominant failure mode in such systems.

**Decision.** Pydantic v2 `BaseModel` (or `BaseSettings` for env vars) at
every boundary. Internal data structures may use `dataclass` or plain
classes; external boundaries must be Pydantic.

**Consequences.**

- Small runtime overhead per validation (Pydantic v2 is C-backed and fast).
- Validation failures at the API edge surface as HTTP `422` with
  field-level error messages, not stack traces.
- Agent tool I/O is validated, so a malformed LLM response triggers a
  retry-or-fallback path rather than crashing the downstream consumer.

---

## ADR-007: Observability primitives written alongside Component 1, not last

**Status:** Accepted (deviates from the original component order).

**Context.** The architecture brief placed observability as Component 7,
implying it is built after the data, models, agent, and API layers. In
practice, instrumentation bolted on at the end is inconsistent: some
modules log at INFO, some at DEBUG, some not at all; latency tracking
covers the API but not the agent steps. The result is exactly the
"observability theatre" reviewers identify quickly.

**Decision.** Write `src/observability/logger.py` and
`src/observability/metrics.py` alongside Component 1 (data pipeline). Every
subsequent component imports and uses them from its first commit.
Component 7 then becomes a polish pass — connecting metrics to a dashboard,
wiring drift signals into alerting — rather than the foundation.

**Consequences.**

- Modest reordering of the build sequence.
- Every component emits structured JSON logs and named latency
  measurements from the moment it exists.
- The eventual dashboard has consistent data to read from instead of
  inconsistent ad-hoc logging.

---

## ADR-008: `hatchling` build backend over `setuptools`

**Status:** Accepted.

**Context.** The architecture brief permitted either `setup.py` or
`pyproject.toml`. The trend across the Python ecosystem since PEP 621
has been toward declarative `pyproject.toml` with PEP 517 build backends.
`setuptools` works but carries historical baggage.

**Decision.** `hatchling` as the build backend, declared in
`[build-system]` of `pyproject.toml`. No `setup.py`.

**Consequences.**

- Fully declarative project metadata; no imperative `setup()` call.
- Faster builds than `setuptools` for typical projects.
- One fewer file at the repo root.

---

## ADR-009: Abstract `MarketDataFetcher` over a single hard-coded provider

**Status:** Accepted.

**Context.** The brief listed "yfinance or Alpha Vantage." yfinance scrapes
Yahoo Finance, is undocumented, and breaks intermittently; Alpha Vantage
has a documented API with rate limits.

**Decision.** Define an abstract `MarketDataFetcher` interface in
`src/data/fetcher.py` with `AlphaVantageFetcher` and `YFinanceFetcher`
implementations. Alpha Vantage is the default for backfill; yfinance is
the fallback when the key is unset or quota is exhausted.

**Consequences.**

- Adds an interface and a strategy hop versus a single hard-coded fetcher.
- The two providers' wire formats are normalised once, in the fetcher
  layer; the rest of the codebase deals only with the canonical
  `OHLCV` schema.
- Adding a third provider (Polygon, dukascopy, broker feed) is one new
  file plus a registry entry.

---

## Template for future ADRs

```
## ADR-NNN: Short imperative title

**Status:** Accepted | Superseded by ADR-XXX | Reversed.

**Context.** The forces at play, the constraints, the alternative
options considered.

**Decision.** What is being chosen, stated as a positive imperative.

**Consequences.** What follows from the decision — costs, benefits, and
the trail of secondary decisions it forces.
```
