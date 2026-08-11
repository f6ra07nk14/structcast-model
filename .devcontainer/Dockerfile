ARG UV_VERSION=latest
FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv
FROM mcr.microsoft.com/devcontainers/base:ubuntu-24.04

ARG DEBIAN_FRONTEND=noninteractive
ARG NVM_VERSION=v0.40.5
ARG NODE_VERSION=20

COPY --from=uv /uv /uvx /usr/local/bin/

ENV UV_CACHE_DIR=/home/vscode/.cache/uv
ENV UV_LINK_MODE=copy
ENV UV_TOOL_BIN_DIR=/home/vscode/.local/bin
ENV CODEX_HOME=/home/vscode/.codex
ENV CODEX_STANDALONE_HOME=/opt/codex
ENV CLAUDE_HOME=/home/vscode/.claude
ENV NVM_DIR=/home/vscode/.nvm
ENV PATH=/home/vscode/.local/bin:/home/vscode/bin:/usr/local/bin:$PATH


USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    bash-completion \
    build-essential \
    ca-certificates \
    curl \
    dnsutils \
    docker-compose-v2 \
    docker.io \
    fd-find \
    fzf \
    git \
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

RUN cat <<'EOF' > /usr/local/bin/docker
#!/usr/bin/env bash
set -euo pipefail

SOCKET_PATH="/var/run/docker.sock"

if [ -S "$SOCKET_PATH" ] && { [ ! -r "$SOCKET_PATH" ] || [ ! -w "$SOCKET_PATH" ]; }; then
  # Preserve HOME so ssh:// docker contexts still use the calling user's
  # ~/.ssh and ~/.docker instead of root's (which has neither).
  exec sudo --preserve-env=HOME /usr/bin/docker "$@"
fi

exec /usr/bin/docker "$@"
EOF

RUN chmod 0755 /usr/local/bin/docker

# The docker wrapper above falls back to `sudo` when the socket isn't
# group-accessible yet (e.g. right after container start, before a shell
# reopen picks up the new docker group). ssh ignores $HOME for its default
# IdentityFile/UserKnownHostsFile lookup and uses root's own passwd entry
# instead, so `docker --context <ssh-context> ...` would otherwise fail
# with "Host key verification failed" when run through that sudo fallback.
# Point ssh at the vscode user's key/known_hosts regardless of which UID
# actually runs it.
RUN mkdir -p /etc/ssh/ssh_config.d && cat <<'EOF' > /etc/ssh/ssh_config.d/60-remote-docker.conf
Host *
    IdentityFile /home/vscode/.ssh/id_ed25519
    IdentityFile /home/vscode/.ssh/id_rsa
    IdentityFile /home/vscode/.ssh/id_ecdsa
    UserKnownHostsFile /home/vscode/.ssh/known_hosts
EOF

RUN mkdir -p /home/vscode/.cache/uv /home/vscode/.codex /home/vscode/.claude /home/vscode/.local/bin /home/vscode/.local/share/uv /commandhistory "$CODEX_STANDALONE_HOME" \
    && chown -R vscode:vscode /home/vscode/.cache /home/vscode/.codex /home/vscode/.claude /home/vscode/.local /commandhistory "$CODEX_STANDALONE_HOME"

USER vscode

RUN curl -fsSL "https://raw.githubusercontent.com/nvm-sh/nvm/${NVM_VERSION}/install.sh" | PROFILE=/dev/null bash

RUN bash -lc ". \"$NVM_DIR/nvm.sh\" && nvm install \"$NODE_VERSION\" && nvm alias default \"$NODE_VERSION\" && nvm use default && corepack enable"

RUN curl -fsSL https://claude.ai/install.sh | bash

# Install the Codex executable payload outside $CODEX_HOME. $CODEX_HOME is
# bind-mounted at runtime for auth/config, so anything installed there during
# image build would be hidden once the container starts.
RUN curl -fsSL https://chatgpt.com/codex/install.sh | CODEX_HOME="$CODEX_STANDALONE_HOME" CODEX_NON_INTERACTIVE=1 sh

RUN curl -fsSL https://raw.githubusercontent.com/rtk-ai/rtk/refs/heads/master/install.sh | sh

RUN bash -lc "command -v codex && codex --version || true"
RUN bash -lc "command -v claude && claude --version"

WORKDIR /workspaces
