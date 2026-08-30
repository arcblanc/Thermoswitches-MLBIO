#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$ROOT"

if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi

python src/de_novo_hallucinations/gener_rna.py --num-samples 2 --temperature 1.0 --top-k 250
python src/validation_embedding/birna_embed.py \
  --input-fasta data/processed/de_novo/smoke/generated.fasta
python scripts/generation/verify_llm_smoke_outputs.py
echo "LLM smoke test passed"
