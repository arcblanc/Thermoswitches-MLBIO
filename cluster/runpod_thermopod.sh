#!/usr/bin/env bash
# Bootstrap + run GenerRNA + BiRNA batch on RunPod Thermopod (inside SSH session).
# Uses RunPod's native PyTorch via --system-site-packages; never pip-installs torch.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ ! -d .venv ]]; then
  python3 -m venv --system-site-packages .venv
fi
source .venv/bin/activate

pip install -q -r requirements-llm.txt -r requirements-aws.txt

echo "=== GPU check (must pass before batch) ==="
python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit(
        "CUDA unavailable. Do not pip install torch on RunPod — use the pytorch template "
        "and requirements-llm.txt without torch."
    )
print(f"torch {torch.__version__} | GPU: {torch.cuda.get_device_name(0)}")
PY

export STORAGE_TARGET="${STORAGE_TARGET:-runpod}"

echo "=== RunPod LLM batch (STORAGE_TARGET=${STORAGE_TARGET}, GENERNA_NUM_SAMPLES=${GENERNA_NUM_SAMPLES:-10000}) ==="
python scripts/generation/llm_cloud_batch.py --dry-run

if [[ "${1:-}" != "--yes" && "${RUNPOD_FULL_BATCH:-}" != "1" ]]; then
  echo ""
  read -r -p "Proceed with full batch? [y/N] " confirm
  if [[ "${confirm,,}" != "y" ]]; then
    echo "Aborted."
    exit 0
  fi
fi

export GENERNA_NUM_SAMPLES="${GENERNA_NUM_SAMPLES:-10000}"
bash scripts/generation/llm_cloud_run.sh
