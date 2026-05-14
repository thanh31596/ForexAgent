# Multi-stage build for ForexAgent API
FROM python:3.11-slim AS builder
WORKDIR /app
RUN pip install --upgrade pip
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip wheel --no-cache-dir -w /wheels .

FROM python:3.11-slim AS runtime
WORKDIR /app
ENV PYTHONUNBUFFERED=1
COPY --from=builder /wheels /wheels
RUN pip install --no-cache /wheels/*.whl && rm -rf /wheels
COPY src ./src
COPY pyproject.toml README.md ./
EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
