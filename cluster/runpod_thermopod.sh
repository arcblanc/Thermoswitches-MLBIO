#!/usr/bin/env bash
# Bootstrap + run GenerRNA + BiRNA batch on RunPod Thermopod (inside SSH session).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi

pip install -q -r requirements-llm.txt -r requirements-aws.txt

export STORAGE_TARGET="${STORAGE_TARGET:-runpod}"

echo "=== RunPod LLM batch (STORAGE_TARGET=${STORAGE_TARGET}, GENERNA_NUM_SAMPLES=${GENERNA_NUM_SAMPLES:-10000}) ==="
python scripts/llm_cloud_batch.py --dry-run

if [[ "${1:-}" != "--yes" && "${RUNPOD_FULL_BATCH:-}" != "1" ]]; then
  echo ""
  read -r -p "Proceed with full batch? [y/N] " confirm
  if [[ "${confirm,,}" != "y" ]]; then
    echo "Aborted."
    exit 0
  fi
fi

export GENERNA_NUM_SAMPLES="${GENERNA_NUM_SAMPLES:-10000}"
bash scripts/llm_cloud_run.sh
