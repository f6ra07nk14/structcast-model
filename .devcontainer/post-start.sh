#!/usr/bin/env bash
set -euo pipefail

SOCKET_PATH="/var/run/docker.sock"

if [ ! -S "$SOCKET_PATH" ]; then
    echo "Docker socket not mounted. Skipping Docker group setup."
    exit 0
fi

SOCKET_GID="$(stat -c '%g' "$SOCKET_PATH")"
SOCKET_GROUP="$(getent group "$SOCKET_GID" | cut -d: -f1 || true)"

if [ -z "$SOCKET_GROUP" ]; then
    SOCKET_GROUP="docker-host"
    sudo groupadd --gid "$SOCKET_GID" "$SOCKET_GROUP"
fi

if ! id -nG "$USER" | tr ' ' '\n' | grep -qx "$SOCKET_GROUP"; then
    sudo usermod -aG "$SOCKET_GROUP" "$USER"
    echo "Added $USER to $SOCKET_GROUP. Rebuild or reopen the shell if Docker is still unavailable."
else
    echo "$USER already has access to $SOCKET_GROUP."
fi
