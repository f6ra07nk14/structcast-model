ARG UV_VERSION=latest
FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv
FROM mcr.microsoft.com/devcontainers/base:ubuntu-24.04

ARG DEBIAN_FRONTEND=noninteractive

COPY --from=uv /uv /uvx /usr/local/bin/

ENV UV_CACHE_DIR=/home/vscode/.cache/uv
ENV UV_LINK_MODE=copy
ENV UV_TOOL_BIN_DIR=/home/vscode/.local/bin
ENV CODEX_HOME=/home/vscode/.codex
ENV CODEX_STANDALONE_HOME=/opt/codex
ENV CLAUDE_HOME=/home/vscode/.claude
ENV PATH=/home/vscode/.local/bin:/home/vscode/bin:/usr/local/bin:$PATH


USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    bash-completion \
    build-essential \
    ca-certificates \
    curl \
    dnsutils \
    fd-find \
    fzf \
    gnupg \
    iproute2 \
    iputils-ping \
    jq \
    less \
    lsb-release \
    nano \
    openssh-client \
    pkg-config \
    python3 \
    python3-pip \
    python3-venv \
    ripgrep \
    sudo \
    unzip \
    vim \
    wget \
    xz-utils \
    zip \
    bubblewrap \
    && rm -rf /var/lib/apt/lists/*

RUN echo "vscode ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/vscode-nopasswd \
    && chmod 0440 /etc/sudoers.d/vscode-nopasswd

RUN mkdir -p /home/vscode/.cache/uv /home/vscode/.codex /home/vscode/.claude /home/vscode/.local/bin /home/vscode/.local/share/uv /commandhistory "$CODEX_STANDALONE_HOME" \
    && chown -R vscode:vscode /home/vscode/.cache /home/vscode/.codex /home/vscode/.claude /home/vscode/.local /commandhistory "$CODEX_STANDALONE_HOME"

USER vscode

RUN curl -fsSL https://claude.ai/install.sh | bash

# Install the Codex executable payload outside $CODEX_HOME. $CODEX_HOME is
# bind-mounted at runtime for auth/config, so anything installed there during
# image build would be hidden once the container starts.
RUN curl -fsSL https://chatgpt.com/codex/install.sh | CODEX_HOME="$CODEX_STANDALONE_HOME" CODEX_NON_INTERACTIVE=1 sh

RUN curl -fsSL https://raw.githubusercontent.com/rtk-ai/rtk/refs/heads/master/install.sh | sh

RUN bash -lc "command -v codex && codex --version || true"
RUN bash -lc "command -v claude && claude --version"

WORKDIR /workspaces
