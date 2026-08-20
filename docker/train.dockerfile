# GPU training image for structcast-model, built for NVIDIA DGX (H200) validation runs.
#
# The image deliberately contains no CUDA toolkit: the uv-locked torch cu130 wheels bundle the
# CUDA 13.0 userspace runtime and NCCL (torch 2.11.0+cu130 declares cuda-toolkit==13.0.2 and
# nvidia-nccl-cu13 as pip dependencies), so the host only needs an NVIDIA driver >= 580.95.05
# and the nvidia container runtime. This validates the exact dependency set in uv.lock instead
# of an NGC container's own PyTorch build.
#
# Build (from the repository root):
#   docker build -f docker/train.dockerfile -t structcast-model-train .
#
# The Flax/JAX variant swaps the framework extra (uv marks the torch and jax extras as
# conflicting, so each framework gets its own image):
#   docker build -f docker/train.dockerfile --build-arg FRAMEWORK_EXTRA=jax-cu12 \
#     -t structcast-model-train-jax .
#
# Run (DGX quickstart + NCCL container guidance; keep the stack ulimit numeric — an unlimited
# stack paradoxically shrinks NCCL's background-thread stacks and crashes communicator setup):
#   docker run --rm -it --gpus '"device=0,1,2,3"' \
#     --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
#     -v "$PWD:/workspace" structcast-model-train \
#     torchrun --standalone --nproc-per-node=4 -m structcast_model.commands.main torch train ...
#
# The repository is bind-mounted at /workspace at run time (PYTHONPATH points at its src/), so
# code changes need no image rebuild; only dependency changes do.

ARG PY_VERSION=3.12
FROM ghcr.io/astral-sh/uv:python${PY_VERSION}-bookworm-slim

# torch.compile needs a host C/C++ toolchain at run time: triton builds its CUDA driver shim and
# inductor its wrapper modules with cc, even though the wheels themselves are binary.
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

ENV HOME=/app
WORKDIR /app
ENV UV_CACHE_DIR=/app/.cache/uv

# Install dependencies only (no project): torch cu130 wheels, experiment tracking, and the dev
# group for running the test suite on the GPUs.
ARG FRAMEWORK_EXTRA=torch-cu130
RUN --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=README.md,target=README.md \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --dev --extra ${FRAMEWORK_EXTRA} --extra mlflow

ENV PATH="/app/.venv/bin:${PATH}"
ENV PYTHONPATH=/workspace/src
WORKDIR /workspace
