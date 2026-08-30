#!/usr/bin/env bash
# Thin wrapper: run EVA cloud batch with active venv.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$ROOT"

if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi

python scripts/eva/eva_cloud_batch.py "$@"
