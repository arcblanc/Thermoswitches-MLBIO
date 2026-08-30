#!/usr/bin/env bash
# Bootstrap + run EVA (smoke or full) + BiRNA on RunPod.
# Prefer an EVA Docker image with eva-generate on PATH.
# On a stock RunPod PyTorch template, install EVA separately before this script.
#
# Usage (inside pod SSH session):
#   bash cluster/runpod_eva_thermopod.sh smoke
#   bash cluster/runpod_eva_thermopod.sh full
#   bash cluster/runpod_eva_thermopod.sh smoke --yes
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MODE="${1:-smoke}"
shift || true

if [[ "${MODE}" != "smoke" && "${MODE}" != "full" ]]; then
  echo "Usage: $0 {smoke|full} [--yes]" >&2
  exit 1
fi

if [[ ! -d .venv ]]; then
  python3 -m venv --system-site-packages .venv
fi
source .venv/bin/activate

pip install -q -r requirements-llm.txt -r requirements-aws.txt

echo "=== GPU check (must pass before EVA batch) ==="
python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit(
        "CUDA unavailable. Use an EVA/PyTorch CUDA template; do not pip-install torch on RunPod."
    )
print(f"torch {torch.__version__} | GPU: {torch.cuda.get_device_name(0)}")
PY

if ! command -v eva-generate >/dev/null 2>&1; then
  echo "WARNING: eva-generate not on PATH."
  echo "  Install EVA (pip install -e /path/to/EVA) or use the official EVA Docker image."
  echo "  Or set EVA_GENERATE_BIN to the full path of eva-generate."
fi

export STORAGE_TARGET="${STORAGE_TARGET:-runpod}"
export AWS_S3_PREFIX="${AWS_S3_PREFIX:-llm-batch/eva/v1}"
export EVA_RNA_TYPE="${EVA_RNA_TYPE:-mRNA}"
export EVA_CHUNK_SIZE="${EVA_CHUNK_SIZE:-512}"
export EVA_CHECKPOINT_DIR="${EVA_CHECKPOINT_DIR:-models/eva/checkpoint}"
export RUNPOD_AUTO_TERMINATE="${RUNPOD_AUTO_TERMINATE:-true}"

if [[ "${MODE}" == "smoke" ]]; then
  export EVA_NUM_SAMPLES="${EVA_NUM_SAMPLES:-16}"
  EXTRA=(--smoke)
  echo "=== EVA SMOKE (STORAGE_TARGET=${STORAGE_TARGET}, prefix=${AWS_S3_PREFIX}) ==="
else
  # Full Option B panel quotas come from env / eva_prompts defaults (10k).
  EXTRA=()
  echo "=== EVA FULL PANEL (STORAGE_TARGET=${STORAGE_TARGET}, prefix=${AWS_S3_PREFIX}) ==="
fi

python scripts/eva/eva_cloud_batch.py "${EXTRA[@]}" --dry-run

YES=0
for arg in "$@"; do
  if [[ "${arg}" == "--yes" ]]; then
    YES=1
  fi
done

if [[ "${YES}" -ne 1 && "${RUNPOD_FULL_BATCH:-}" != "1" ]]; then
  echo ""
  read -r -p "Proceed with EVA ${MODE} batch? [y/N] " confirm
  if [[ "${confirm,,}" != "y" ]]; then
    echo "Aborted."
    exit 0
  fi
fi

if [[ ! -d "${EVA_CHECKPOINT_DIR}" ]]; then
  echo "Checkpoint missing at ${EVA_CHECKPOINT_DIR}"
  echo "Download with:"
  echo "  huggingface-cli download GENTEL-Lab/EVA --local-dir ${EVA_CHECKPOINT_DIR}"
  exit 1
fi

bash scripts/eva/eva_cloud_run.sh "${EXTRA[@]}"
