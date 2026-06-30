#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi

python src/de_novo_hallucinations/gener_rna.py --num-samples 2 --temperature 1.0 --top-k 250
python src/validation_embedding/birna_embed.py \
  --input-fasta data/processed/de_novo/smoke/generated.fasta
python scripts/verify_llm_smoke_outputs.py
echo "LLM smoke test passed"
