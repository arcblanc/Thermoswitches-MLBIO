#!/usr/bin/env bash
# Local / pod one-liner: EVA smoke → BiRNA → verify.
# Prefers orchestrator (handles pod stop). Falls back to stepwise if needed.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$ROOT"

if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi

export EVA_RNA_TYPE="${EVA_RNA_TYPE:-mRNA}"
export EVA_NUM_SAMPLES="${EVA_NUM_SAMPLES:-16}"
export EVA_CHUNK_SIZE="${EVA_CHUNK_SIZE:-512}"
# Keep EVA artifacts off GenerRNA prefix when remote
if [[ "${STORAGE_TARGET:-local}" == "runpod" || "${STORAGE_TARGET:-local}" == "s3" ]]; then
  export AWS_S3_PREFIX="${AWS_S3_PREFIX:-llm-batch/eva/v1}"
fi

echo "=== EVA smoke dry-run ==="
python scripts/eva/eva_cloud_batch.py --smoke --dry-run

echo "=== EVA smoke + BiRNA + verify ==="
python scripts/eva/eva_cloud_batch.py --smoke

echo "EVA smoke test passed"
