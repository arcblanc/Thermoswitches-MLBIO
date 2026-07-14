#!/usr/bin/env bash
# Thin wrapper: run EVA cloud batch with active venv.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi

python scripts/eva_cloud_batch.py "$@"
