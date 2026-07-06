#!/usr/bin/env bash
# Resume Mac thermo full pipeline without clearing existing outputs.
# Prefer launching via:
#   python scripts/launch_detached.py --log data/processed/mac_thermo_full.log -- \
#     bash scripts/mac_thermo_resume.sh
# so Mac sleep / terminal close does not kill the job.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source .venv/bin/activate
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

# --isolate-subprocess (default): recycle each worker after one sequence to release
# ViennaRNA/NUPACK C-extension memory to the OS.
COMMON=(
  --run --resume
  --workers 4 --batch-size 4
  --temps 37,41,45,49,53,55
  --sodium 0.05 --magnesium 0.0
  --isolate-subprocess
)

echo "=== RESUME Rfam balanced thermo $(date -u) ==="
python src/thermo_sim/thermo_batch.py "${COMMON[@]}" \
  --input-mode balanced --limit 2396 \
  --vienna-csv data/processed/viennarna/features.csv \
  --nupack-csv data/processed/nupack/features.csv \
  --fused-csv data/processed/fused_features.csv

echo "=== Train RF $(date -u) ==="
python src/thermo_sim/thermo_classifier.py train \
  --fused-csv data/processed/fused_features.csv \
  --model-path data/processed/models/rf_thermoswitch.joblib

echo "=== De novo thermo $(date -u) ==="
python src/thermo_sim/thermo_batch.py "${COMMON[@]}" \
  --input-mode fasta \
  --input-fasta data/processed/de_novo/generated.fasta \
  --limit 10000 \
  --vienna-csv data/processed/viennarna/denovo_features.csv \
  --nupack-csv data/processed/nupack/denovo_features.csv \
  --fused-csv data/processed/denovo_fused_features.csv

echo "=== Predict $(date -u) ==="
python src/thermo_sim/thermo_classifier.py predict \
  --fused-csv data/processed/denovo_fused_features.csv \
  --model-path data/processed/models/rf_thermoswitch.joblib \
  --predictions-csv data/processed/denovo_predictions.csv

echo "=== COMPLETE $(date -u) ==="
wc -l data/processed/fused_features.csv data/processed/denovo_fused_features.csv data/processed/denovo_predictions.csv
