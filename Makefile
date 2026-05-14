# ---------------------------------------------------------------------------
# ForexAgent — Makefile
#
# Run `make help` for a list of targets.
# ---------------------------------------------------------------------------

.PHONY: help \
        setup install dev-install clean lock \
        format lint typecheck pre-commit check \
        test test-fast test-cov \
        train predict evaluate serve mlflow-ui \
        docker-build docker-up docker-down docker-logs \
        ci

# Prefer 3.11+ when available (pyproject requires >=3.11); fall back to python3.
PYTHON      := $(shell command -v python3.11 >/dev/null 2>&1 && echo python3.11 || echo python3)
VENV        := .venv
BIN         := $(VENV)/bin
SRC_DIRS    := src tests
FLAKE_FLAGS := --max-line-length=100 --extend-ignore=E203,W503 --exclude=.venv,build,dist,mlruns

.DEFAULT_GOAL := help

# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------
help:  ## Show this help
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
setup: $(VENV)/.touchfile  ## Create venv, install dev deps, install pre-commit

$(VENV)/.touchfile:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -e ".[dev]"
	$(BIN)/pre-commit install || true
	@touch $(VENV)/.touchfile

install:  ## Install production dependencies only
	$(BIN)/pip install -e ".[prod]"

dev-install:  ## Install dev dependencies into existing venv
	$(BIN)/pip install -e ".[dev]"

lock:  ## Regenerate requirements.txt from pyproject.toml (pinned)
	$(BIN)/uv pip compile pyproject.toml -o requirements.txt
	$(BIN)/uv pip compile pyproject.toml --extra dev -o requirements-dev.txt

clean:  ## Remove build artifacts and caches
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache .coverage coverage.xml htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# ---------------------------------------------------------------------------
# Code quality
# ---------------------------------------------------------------------------
format:  ## Auto-format with black and isort
	$(BIN)/black $(SRC_DIRS)
	$(BIN)/isort $(SRC_DIRS)

lint:  ## Run flake8 (black-compatible rules)
	$(BIN)/flake8 $(SRC_DIRS) $(FLAKE_FLAGS)

typecheck:  ## Run mypy --strict
	$(BIN)/mypy src

pre-commit:  ## Run all pre-commit hooks
	$(BIN)/pre-commit run --all-files

check: format lint typecheck  ## Run all code-quality checks (pre-push gate)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
test:  ## Run all tests with coverage
	$(BIN)/pytest

test-fast:  ## Run unit tests only (skip slow + integration markers)
	$(BIN)/pytest -m "not slow and not integration"

test-cov:  ## Run tests and emit HTML coverage report
	$(BIN)/pytest --cov-report=html
	@echo "Coverage report: htmlcov/index.html"

# ---------------------------------------------------------------------------
# Pipeline operations
# ---------------------------------------------------------------------------
train:  ## Train models and register best in MLflow
	$(BIN)/python -m src.models.trainer

predict:  ## Run a sample end-to-end prediction (data + model + agent)
	$(BIN)/python -m src.api.main --predict-sample

evaluate:  ## Run the evaluation framework against baselines
	$(BIN)/python -m src.evaluation.evaluator

serve:  ## Start FastAPI service on :8000 with hot reload
	$(BIN)/uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

mlflow-ui:  ## Launch MLflow UI on :5000 against local store
	$(BIN)/mlflow ui --backend-store-uri ./mlruns --host 0.0.0.0 --port 5000

# ---------------------------------------------------------------------------
# Docker
# ---------------------------------------------------------------------------
docker-build:  ## Build the production image
	docker build -f Dockerfile -t forexagent:latest .

docker-up:  ## Start full stack (api + postgres + mlflow + minio)
	docker compose up -d

docker-down:  ## Stop full stack
	docker compose down

docker-logs:  ## Tail compose logs
	docker compose logs -f

# ---------------------------------------------------------------------------
# CI helper
# ---------------------------------------------------------------------------
ci: lint typecheck test  ## Run the full CI suite locally
