# ================================================================
# IX03 — Multi-Architecture pyquantflow Container
#
# Targets:
#   quant-engine       : lean headless compute (GCP Cloud Run / amd64)
#   quant-orchestrator : edge execution with Prefect (Raspberry Pi 5 / arm64)
#
# Build:
#   docker buildx build --target engine       -t quant-engine:local .
#   docker buildx build --target orchestrator -t quant-orchestrator:local .
# ================================================================

# ----------------------------------------------------------------
# Stage 1: base
# Installs uv and system build tools.
# This layer is never shipped; it exists solely to cache uv.
# ----------------------------------------------------------------
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_SYSTEM_PYTHON=0

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

# Install uv via the official installer script
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

ENV PATH="/root/.local/bin:$PATH"

# ----------------------------------------------------------------
# Stage 2: builder
# Resolves and installs all runtime dependencies into /app/.venv
# using the frozen uv.lock for fully reproducible builds.
# ----------------------------------------------------------------
FROM base AS builder

WORKDIR /app

# Copy lockfiles first — this layer is cache-invalidated only when
# pyproject.toml or uv.lock change, not on source code changes.
COPY pyproject.toml uv.lock README.md ./
COPY pyquantflow/ ./pyquantflow/

# --frozen: enforce exact lockfile; --no-dev: exclude dev dependencies
RUN uv sync --frozen --no-dev

# ----------------------------------------------------------------
# Stage 3: engine  (target: quant-engine)
# Sterile final image. Copies only the pre-built venv and source.
# Executes under a non-root user (appuser:appgroup, UID/GID 1001).
# ----------------------------------------------------------------
FROM python:3.12-slim AS engine

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Copy the compiled virtual environment and package source only
COPY --from=builder /app/.venv      /app/.venv
COPY --from=builder /app/pyquantflow /app/pyquantflow

# Create non-root user and group
RUN groupadd --gid 1001 appgroup \
    && useradd  --uid 1001 \
                --gid 1001 \
                --no-create-home \
                --shell /sbin/nologin \
                appuser

USER appuser

# Default: verify import. Override CMD at runtime for actual workloads.
CMD ["python", "-c", "import pyquantflow; print('pyquantflow', pyquantflow.__version__)"]

# ----------------------------------------------------------------
# Stage 4: orchestrator  (target: quant-orchestrator)
# Inherits the sterile engine image and adds Prefect for edge
# workflow orchestration. uv is brought in temporarily to install
# Prefect into the existing venv, then removed.
# ----------------------------------------------------------------
FROM engine AS orchestrator

# Temporarily elevate to root to install Prefect
USER root

# Copy uv binary from the base stage (avoids re-downloading)
COPY --from=base /root/.local/bin/uv /usr/local/bin/uv

# Install Prefect into the already-activated venv, then purge uv
RUN uv pip install --python /app/.venv/bin/python prefect \
    && rm /usr/local/bin/uv

# Return to non-root user (inherited from engine)
USER appuser

# Default: start a Prefect worker. Override at runtime as needed.
CMD ["prefect", "worker", "start", "--pool", "default-agent-pool"]
