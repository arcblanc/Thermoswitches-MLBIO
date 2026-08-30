#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$ROOT"

if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi

python3 -m venv /tmp/thermo-llm-test
source /tmp/thermo-llm-test/bin/activate
pip install -q -r requirements-llm.txt
python -c "import torch; import transformers; print('torch', torch.__version__); print('transformers', transformers.__version__)"
