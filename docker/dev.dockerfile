ARG PY_VERSION=3.11
FROM ghcr.io/astral-sh/uv:python${PY_VERSION}-bookworm-slim

# Install build dependencies for compiling Python packages and Node.js
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Define Python versions variable
# Note: Python 3.14 is excluded as it's not yet supported by pydantic-core/PyO3
ARG PYTHON_VERSIONS="3.11 3.12 3.13"

# Set working directory
ENV HOME=/app
WORKDIR /app

# Configure UV cache directory (persistent across HOME changes)
ENV UV_CACHE_DIR=/app/.cache/uv

# Install multiple Python versions
RUN uv python install $PYTHON_VERSIONS --preview

# Install dependencies using uv sync (recommended method)
# The venv created in /app/.venv will be reused in CI via PATH
RUN --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=README.md,target=README.md \
    --mount=type=bind,source=tox.ini,target=tox.ini \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --dev --group tox --extra torch-cpu --extra flops

# Activate virtual environment and set up PATH to use installed tools
ENV PATH="/app/.venv/bin:${PATH}"

# Set Python path
ENV PYTHONPATH=/app/src

