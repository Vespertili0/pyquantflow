# ================================================================
# Multi-Architecture pyquantflow Container
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
# Sources the uv binary directly from the official astral-sh OCI
# image — no curl|sh, no unverified remote script execution.
# ----------------------------------------------------------------
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_SYSTEM_PYTHON=0

# Supply-chain safe: copy uv from the pinned upstream image layer
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

# ----------------------------------------------------------------
# Stage 2: builder
# Two-layer cache strategy separates expensive dependency resolution
# from cheap source-code changes:
#
#   Layer A  →  installs all deps (no project); cache-stable across
#               every commit that doesn't touch pyproject.toml/uv.lock
#   Layer B  →  installs pyquantflow itself; only invalidated when
#               source files under pyquantflow/ change
# ----------------------------------------------------------------
FROM base AS builder

WORKDIR /app

# ── Layer A: resolve and install dependencies only ───────────────
COPY pyproject.toml uv.lock README.md ./

# --no-install-project: skip pyquantflow itself so this expensive
# layer is fully reused on source-only commits.
RUN uv sync --frozen --no-install-project --no-dev

# ── Layer B: copy source, then install the package ───────────────
COPY pyquantflow/ ./pyquantflow/

# Full sync: installs pyquantflow into the already-populated venv.
RUN uv sync --frozen --no-dev

# ----------------------------------------------------------------
# Stage 3: orchestrator-builder
# Extends builder with the locked `orchestrator` optional dependency
# group (Prefect). All transitive versions are resolved from
# uv.lock at lock-time — no dynamic resolution at build-time.
# ----------------------------------------------------------------
FROM builder AS orchestrator-builder

RUN uv sync --frozen --extra orchestrator --no-dev

# ----------------------------------------------------------------
# Stage 4: runtime-base
# Shared foundation for all runtime targets.
# Non-root user is created before COPY so --chown resolves at layer
# creation time, giving appuser ownership of all runtime files.
# ----------------------------------------------------------------
FROM python:3.12-slim AS runtime-base

# Create non-root user before any COPY so --chown can resolve them
RUN groupadd --gid 1001 appgroup \
    && useradd  --uid 1001 \
                --gid 1001 \
                --no-create-home \
                --shell /sbin/nologin \
                appuser

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH"

USER appuser

# ----------------------------------------------------------------
# Stage 5: engine  (target: quant-engine)
# Sterile final image for headless compute workloads.
# ----------------------------------------------------------------
FROM runtime-base AS engine

# --chown: appuser owns all venv and source files; no PermissionError
# if the application writes to paths relative to the working directory
# via an explicitly mounted volume.
COPY --from=builder --chown=appuser:appgroup /app/.venv      /app/.venv
COPY --from=builder --chown=appuser:appgroup /app/pyquantflow /app/pyquantflow

# Default: verify import. Override CMD at runtime for actual workloads.
CMD ["python", "-c", "import pyquantflow; print('pyquantflow', pyquantflow.__version__)"]

# ----------------------------------------------------------------
# Stage 6: orchestrator  (target: quant-orchestrator)
# Inherits the shared runtime base and copies the orchestrator-specific
# venv (with locked Prefect). This avoids dirty merges with the engine's
# venv that would break bit-for-bit deployment guarantees.
# ----------------------------------------------------------------
FROM runtime-base AS orchestrator

COPY --from=orchestrator-builder --chown=appuser:appgroup /app/.venv /app/.venv
COPY --from=builder --chown=appuser:appgroup /app/pyquantflow /app/pyquantflow

CMD ["prefect", "worker", "start", "--pool", "default-agent-pool"]
