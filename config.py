"""Application configuration.

All runtime configuration for ForexAgent lives in this module. Two concerns
are separated:

* **Module-level constants** — domain enums, indicator windows, model
  hyperparameters, evaluation thresholds, file paths. These are *code*, not
  environment; changing them changes the project's behaviour and should be a
  versioned commit.

* **The :class:`Settings` model** — runtime settings loaded from environment
  variables or a ``.env`` file via :mod:`pydantic_settings`. These are
  *secrets and deployment knobs*: API keys, database URLs, log levels, the
  environment label.

Every magic number referenced elsewhere in the codebase MUST be declared
here under a named constant. Inline literals are a code-review failure
mode.
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, computed_field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# Filesystem layout
# ---------------------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
MODEL_DIR: Path = PROJECT_ROOT / "models_artifacts"
LOG_DIR: Path = PROJECT_ROOT / "logs"
METRICS_DIR: Path = PROJECT_ROOT / "metrics"
MLFLOW_LOCAL_STORE: Path = PROJECT_ROOT / "mlruns"

for _dir in (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    MODEL_DIR,
    LOG_DIR,
    METRICS_DIR,
):
    _dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Domain enums
# ---------------------------------------------------------------------------

class CurrencyPair(str, Enum):
    """Supported FX pairs (yfinance ticker convention).

    The string value is the symbol passed to the fetcher; Alpha Vantage uses
    a different convention which the fetcher layer maps internally.
    """

    EUR_USD = "EURUSD=X"
    GBP_USD = "GBPUSD=X"
    USD_JPY = "USDJPY=X"


SUPPORTED_PAIRS: tuple[CurrencyPair, ...] = tuple(CurrencyPair)


class DataInterval(str, Enum):
    """Bar interval for candlestick data."""

    ONE_HOUR = "1h"
    FOUR_HOUR = "4h"
    ONE_DAY = "1d"


class TradingSignal(str, Enum):
    """Discrete trading action emitted by the agent."""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class LogLevel(str, Enum):
    """Standard Python logging levels exposed as an enum for type safety."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


# ---------------------------------------------------------------------------
# Feature engineering defaults  (Component 1 — Data)
# ---------------------------------------------------------------------------

# Indicator window sizes (in bars).
RSI_WINDOW: int = 14
MACD_FAST_WINDOW: int = 12
MACD_SLOW_WINDOW: int = 26
MACD_SIGNAL_WINDOW: int = 9
BOLLINGER_WINDOW: int = 20
BOLLINGER_STD: float = 2.0
SMA_WINDOWS: tuple[int, ...] = (5, 10, 20, 50, 200)
EMA_WINDOWS: tuple[int, ...] = (5, 10, 20, 50)
VOLATILITY_WINDOW: int = 20
ATR_WINDOW: int = 14
RETURN_LAGS: tuple[int, ...] = (1, 2, 3, 5, 10)

# Target definition: predict log-return ``PREDICTION_HORIZON`` bars ahead.
PREDICTION_HORIZON: int = 1
TARGET_TYPE: Literal["log_return", "direction", "price"] = "log_return"

# Default backfill window for historical data, in days.
DEFAULT_BACKFILL_DAYS: int = 730  # ~ 2 years of daily bars


# ---------------------------------------------------------------------------
# Training defaults  (Component 2 — Classical ML)
# ---------------------------------------------------------------------------

# Walk-forward CV configuration.
WALK_FORWARD_N_SPLITS: int = 5
WALK_FORWARD_TEST_SIZE: int = 200  # bars per test fold
WALK_FORWARD_GAP: int = 1          # bars between train end and test start
WALK_FORWARD_STRATEGY: Literal["expanding", "sliding"] = "expanding"

# Reproducibility.
RANDOM_SEED: int = 42

# LightGBM defaults. Override per-experiment via MLflow runs.
LGBM_PARAMS: dict[str, float | int | str] = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "min_data_in_leaf": 20,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "seed": RANDOM_SEED,
}
LGBM_NUM_BOOST_ROUND: int = 1000
LGBM_EARLY_STOPPING_ROUNDS: int = 50

# PyTorch LSTM defaults.
LSTM_SEQUENCE_LENGTH: int = 60
LSTM_HIDDEN_SIZE: int = 64
LSTM_NUM_LAYERS: int = 2
LSTM_DROPOUT: float = 0.2
LSTM_LEARNING_RATE: float = 1e-3
LSTM_BATCH_SIZE: int = 64
LSTM_EPOCHS: int = 50
LSTM_PATIENCE: int = 10  # early-stopping patience (epochs)

# MLflow.
MLFLOW_EXPERIMENT_NAME: str = "forexagent"
MLFLOW_REGISTERED_MODEL_NAME: str = "forexagent-classical"
MLFLOW_STAGE_PRODUCTION: str = "Production"
MLFLOW_STAGE_STAGING: str = "Staging"


# ---------------------------------------------------------------------------
# Agent defaults  (Component 3 — Agentic AI)
# ---------------------------------------------------------------------------

AGENT_MODEL: str = "gpt-4o-mini"
AGENT_TEMPERATURE: float = 0.0
AGENT_MAX_TOKENS: int = 1024
AGENT_MAX_ITERATIONS: int = 5
AGENT_TIMEOUT_S: float = 30.0

# RAG configuration.
NEWS_RETRIEVAL_TOP_K: int = 5
NEWS_LOOKBACK_HOURS: int = 24
RAG_CHUNK_SIZE: int = 512
RAG_CHUNK_OVERLAP: int = 64
RAG_EMBEDDING_MODEL: str = "text-embedding-3-small"
RAG_COLLECTION_NAME: str = "forex_news"

# Signal-generation thresholds (probability of upward move).
SIGNAL_BUY_THRESHOLD: float = 0.55      # P(up) > this -> BUY
SIGNAL_SELL_THRESHOLD: float = 0.45     # P(up) < this -> SELL
# Anything between the two -> HOLD.


# ---------------------------------------------------------------------------
# Evaluation defaults  (Component 4 — Evaluation)
# ---------------------------------------------------------------------------

EVAL_ROLLING_WINDOW: int = 50  # predictions per rolling-metric calculation
EVAL_BASELINE_NAMES: tuple[str, ...] = (
    "buy_and_hold",
    "ma_crossover",
    "random",
)
EVAL_TRADING_COST_BPS: float = 1.0          # 1 basis point per round-trip
EVAL_RISK_FREE_RATE_ANNUAL: float = 0.04    # for Sharpe
EVAL_TRADING_DAYS_PER_YEAR: int = 252

# Drift detection.
DRIFT_PSI_THRESHOLD: float = 0.2            # PSI > 0.2 -> significant drift
DRIFT_SHARPE_DEGRADATION_PCT: float = 0.5   # 50% drop in rolling Sharpe -> drift
DRIFT_LOOKBACK_WINDOW: int = 200            # bars used as drift reference


# ---------------------------------------------------------------------------
# API defaults  (Component 5 — API)
# ---------------------------------------------------------------------------

API_HOST: str = "0.0.0.0"
API_PORT: int = 8000
API_TITLE: str = "ForexAgent API"
API_VERSION: str = "0.1.0"
API_DESCRIPTION: str = (
    "Forex trading-signal service combining a classical ML pipeline with an "
    "agentic AI reasoning layer."
)
API_REQUEST_TIMEOUT_S: float = 30.0
API_RATE_LIMIT_PER_MINUTE: int = 60


# ---------------------------------------------------------------------------
# Observability defaults  (Component 7 — Observability)
# ---------------------------------------------------------------------------

LOG_FORMAT_JSON: str = (
    "%(asctime)s %(levelname)s %(name)s %(module)s "
    "%(funcName)s %(lineno)d %(message)s"
)
LOG_FILENAME: str = "forexagent.log"
LATENCY_METRICS_FILENAME: str = "latency.jsonl"
EVAL_METRICS_FILENAME: str = "evaluations.jsonl"
PREDICTION_LOG_FILENAME: str = "predictions.jsonl"


# ---------------------------------------------------------------------------
# Settings (env-driven)
# ---------------------------------------------------------------------------

class Settings(BaseSettings):
    """Runtime settings loaded from environment variables or a ``.env`` file.

    Instantiated once at import time via :func:`get_settings`, and exposed
    as the module-level :data:`settings` singleton. For FastAPI, prefer
    injecting via ``Depends(get_settings)`` so tests can override.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Environment ---
    env: Literal["dev", "staging", "prod"] = Field(
        default="dev",
        description="Runtime environment label.",
    )

    # --- Logging ---
    log_level: LogLevel = Field(
        default=LogLevel.INFO,
        description="Root logger level.",
    )
    log_to_file: bool = Field(
        default=True,
        description="If True, also write logs to LOG_DIR/forexagent.log.",
    )

    # --- Market data ---
    alpha_vantage_api_key: SecretStr | None = Field(
        default=None,
        description="Alpha Vantage API key. Falls back to yfinance if absent.",
    )

    # --- News / RAG ---
    marketaux_api_key: SecretStr | None = Field(
        default=None,
        description="Marketaux API key for FX news retrieval.",
    )
    finnhub_api_key: SecretStr | None = Field(
        default=None,
        description="Optional alternative news provider.",
    )

    # --- LLM ---
    openai_api_key: SecretStr | None = Field(
        default=None,
        description="OpenAI API key for the agent backend.",
    )
    agent_model_override: str | None = Field(
        default=None,
        description="If set, overrides AGENT_MODEL at runtime.",
    )

    # --- Database ---
    database_url: str = Field(
        default=f"sqlite:///{PROJECT_ROOT / 'forexagent.db'}",
        description="SQLAlchemy DSN. SQLite for dev, Postgres for compose.",
    )

    # --- MLflow ---
    mlflow_tracking_uri: str = Field(
        default=f"file://{MLFLOW_LOCAL_STORE}",
        description="MLflow tracking URI. Use http://mlflow:5000 in compose.",
    )

    # --- API ---
    api_host: str = Field(default=API_HOST)
    api_port: int = Field(default=API_PORT)

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------
    @field_validator("database_url")
    @classmethod
    def _validate_database_url(cls, v: str) -> str:
        """Reject unsupported DB schemes early."""
        allowed_prefixes = (
            "sqlite:///",
            "postgresql://",
            "postgresql+psycopg2://",
        )
        if not v.startswith(allowed_prefixes):
            raise ValueError(
                "DATABASE_URL must start with one of: "
                f"{', '.join(allowed_prefixes)}"
            )
        return v

    # ------------------------------------------------------------------
    # Computed properties
    # ------------------------------------------------------------------
    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_production(self) -> bool:
        """True if the runtime environment is production."""
        return self.env == "prod"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_using_postgres(self) -> bool:
        """True if the database is PostgreSQL (vs. SQLite)."""
        return self.database_url.startswith("postgresql")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def agent_model(self) -> str:
        """Resolved agent model name, honouring runtime override."""
        return self.agent_model_override or AGENT_MODEL


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the singleton :class:`Settings` instance.

    Cached so env parsing happens once per process. Suitable as a FastAPI
    dependency::

        from fastapi import Depends
        from src.config import Settings, get_settings

        @app.get("/health")
        def health(s: Settings = Depends(get_settings)) -> dict:
            return {"env": s.env}

    Tests can override by calling ``get_settings.cache_clear()`` and
    re-importing, or by overriding the dependency in the FastAPI app.

    Returns
    -------
    Settings
        The cached, validated settings instance.
    """
    return Settings()


# Module-level handle for non-DI call sites.
settings: Settings = get_settings()


__all__ = [
    # Enums
    "CurrencyPair",
    "DataInterval",
    "TradingSignal",
    "LogLevel",
    # Settings
    "Settings",
    "settings",
    "get_settings",
    # Paths
    "PROJECT_ROOT",
    "DATA_DIR",
    "RAW_DATA_DIR",
    "PROCESSED_DATA_DIR",
    "MODEL_DIR",
    "LOG_DIR",
    "METRICS_DIR",
    "MLFLOW_LOCAL_STORE",
    # Pairs
    "SUPPORTED_PAIRS",
    # Feature engineering
    "RSI_WINDOW",
    "MACD_FAST_WINDOW",
    "MACD_SLOW_WINDOW",
    "MACD_SIGNAL_WINDOW",
    "BOLLINGER_WINDOW",
    "BOLLINGER_STD",
    "SMA_WINDOWS",
    "EMA_WINDOWS",
    "VOLATILITY_WINDOW",
    "ATR_WINDOW",
    "RETURN_LAGS",
    "PREDICTION_HORIZON",
    "TARGET_TYPE",
    "DEFAULT_BACKFILL_DAYS",
    # Training
    "WALK_FORWARD_N_SPLITS",
    "WALK_FORWARD_TEST_SIZE",
    "WALK_FORWARD_GAP",
    "WALK_FORWARD_STRATEGY",
    "RANDOM_SEED",
    "LGBM_PARAMS",
    "LGBM_NUM_BOOST_ROUND",
    "LGBM_EARLY_STOPPING_ROUNDS",
    "LSTM_SEQUENCE_LENGTH",
    "LSTM_HIDDEN_SIZE",
    "LSTM_NUM_LAYERS",
    "LSTM_DROPOUT",
    "LSTM_LEARNING_RATE",
    "LSTM_BATCH_SIZE",
    "LSTM_EPOCHS",
    "LSTM_PATIENCE",
    # MLflow
    "MLFLOW_EXPERIMENT_NAME",
    "MLFLOW_REGISTERED_MODEL_NAME",
    "MLFLOW_STAGE_PRODUCTION",
    "MLFLOW_STAGE_STAGING",
    # Agent
    "AGENT_MODEL",
    "AGENT_TEMPERATURE",
    "AGENT_MAX_TOKENS",
    "AGENT_MAX_ITERATIONS",
    "AGENT_TIMEOUT_S",
    "NEWS_RETRIEVAL_TOP_K",
    "NEWS_LOOKBACK_HOURS",
    "RAG_CHUNK_SIZE",
    "RAG_CHUNK_OVERLAP",
    "RAG_EMBEDDING_MODEL",
    "RAG_COLLECTION_NAME",
    "SIGNAL_BUY_THRESHOLD",
    "SIGNAL_SELL_THRESHOLD",
    # Evaluation
    "EVAL_ROLLING_WINDOW",
    "EVAL_BASELINE_NAMES",
    "EVAL_TRADING_COST_BPS",
    "EVAL_RISK_FREE_RATE_ANNUAL",
    "EVAL_TRADING_DAYS_PER_YEAR",
    "DRIFT_PSI_THRESHOLD",
    "DRIFT_SHARPE_DEGRADATION_PCT",
    "DRIFT_LOOKBACK_WINDOW",
    # API
    "API_HOST",
    "API_PORT",
    "API_TITLE",
    "API_VERSION",
    "API_DESCRIPTION",
    "API_REQUEST_TIMEOUT_S",
    "API_RATE_LIMIT_PER_MINUTE",
    # Observability
    "LOG_FORMAT_JSON",
    "LOG_FILENAME",
    "LATENCY_METRICS_FILENAME",
    "EVAL_METRICS_FILENAME",
    "PREDICTION_LOG_FILENAME",
]
