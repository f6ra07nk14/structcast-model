ARG PY_VERSION=3.11
FROM ghcr.io/astral-sh/uv:python${PY_VERSION}-bookworm-slim

# Install build dependencies
# Note: build-essential is not needed for binary wheel installs (torch-cpu, etc.)
# curl is required by uv python install --preview to download Python binaries
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
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
# One sync per interpreter warms the uv cache with the cp311/cp312/cp313 wheels the py{311,312,313}
# tox legs need (a single sync would leave two of them downloading at test time), and the wandb
# extra matches the tox default env. 3.11 runs last so /app/.venv keeps the image's default
# interpreter. Keep the list in step with PYTHON_VERSIONS above.
RUN --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=README.md,target=README.md \
    --mount=type=bind,source=tox.ini,target=tox.ini \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    for version in 3.13 3.12 3.11; do \
    uv sync --frozen --no-install-project --python $version \
    --dev --group tox --extra all-cpu --extra mlflow --extra flops --extra wandb \
    || exit 1; \
    done

# Activate virtual environment and set up PATH to use installed tools
ENV PATH="/app/.venv/bin:${PATH}"

# Set Python path
ENV PYTHONPATH=/app/src

