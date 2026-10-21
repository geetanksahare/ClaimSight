# ClaimSight Production Multi-Stage Dockerfile
# Stage 1: Build & Dependencies
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.lock pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.lock

# Stage 2: Runtime Environment
FROM python:3.11-slim AS runner

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Create unprivileged application user
RUN groupadd -r claimsight && useradd -r -g claimsight -d /app -s /sbin/nologin claimsight

# Copy application code, configs, demo data, models, and reports
COPY configs/ /app/configs/
COPY data/demo/ /app/data/demo/
COPY models/ /app/models/
COPY reports/ /app/reports/
COPY src/ /app/src/
COPY pyproject.toml /app/

# Install claimsight in editable/current mode without dependencies
RUN pip install --no-cache-dir --no-deps -e .

RUN chown -R claimsight:claimsight /app

USER claimsight

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CLAIMSIGHT_ENV=production \
    MLFLOW_ALLOW_FILE_STORE=true

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || curl -f http://localhost:8501/_stcore/health || exit 1

# Default command: launch FastAPI service (override with streamlit for UI)
CMD ["uvicorn", "claimsight.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
