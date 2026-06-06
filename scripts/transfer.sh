#!/usr/bin/env bash
# Transfer this project to a remote machine via rsync (uses ~/.ssh/config, so ProxyJump works).
# Repeatable for any box in cuda_machines_info — just pass its SSH alias.
#
# Usage: scripts/transfer.sh <ssh-host> [remote-dir]
#   <ssh-host>    SSH alias (e.g. sie271-pc, sie236)
#   [remote-dir]  path on the remote, relative to home (default: kernels)
#
# Transfers source + pyproject + secrets.env (for HF_TOKEN). Excludes the venv, git,
# caches, and generated artifacts. Does NOT install deps — the env is per-machine
# (see cuda_machines_info); on the Spark, build on ~/envs/test_nightly, e.g.:
#   uv pip install --python ~/envs/test_nightly/bin/python -e <remote-dir> --no-deps
#   uv pip install --python ~/envs/test_nightly/bin/python transformers tabulate tqdm rich ninja
set -euo pipefail

HOST="${1:?usage: scripts/transfer.sh <ssh-host> [remote-dir]}"
REMOTE_DIR="${2:-kernels}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

ssh "$HOST" "mkdir -p '$REMOTE_DIR'"
rsync -avz --delete \
  --exclude '.git/' \
  --exclude '.venv/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '.ruff_cache/' \
  --exclude 'analysis/' \
  --exclude '*.parquet' \
  "$ROOT/" "$HOST:$REMOTE_DIR/"

echo "Transferred $ROOT -> $HOST:$REMOTE_DIR"
