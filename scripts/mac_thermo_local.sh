#!/usr/bin/env bash
# Mac M3 local thermo: Rfam balanced train + de novo 10k predict.
# Usage:
#   bash scripts/mac_thermo_local.sh smoke   # 4 sequences end-to-end
#   bash scripts/mac_thermo_local.sh full    # 2396 train + 10000 de novo
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MODE="${1:-smoke}"
WORKERS="${WORKERS:-4}"
BATCH_SIZE="${BATCH_SIZE:-4}"
# Include 37/45/55 plus neighbors so Hill/Tm fits are stable (3 points alone often fails).
TEMPS="${TEMPS:-37,41,45,49,53,55}"
SODIUM="${SODIUM:-0.05}"
# NUPACK RNA parameter sets do not support Mg salt correction (max=0).
MAGNESIUM="${MAGNESIUM:-0.0}"
S3_BUCKET="${AWS_S3_BUCKET:-thermo-s3-bucket}"
S3_PREFIX="${AWS_S3_PREFIX:-llm-batch/v1}"

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

COMMON_FLAGS=(
  --run
  --resume
  --workers "$WORKERS"
  --batch-size "$BATCH_SIZE"
  --temps "$TEMPS"
  --sodium "$SODIUM"
  --magnesium "$MAGNESIUM"
  --isolate-subprocess
)

pull_denovo_fasta() {
  mkdir -p data/processed/de_novo
  if [[ -f data/processed/de_novo/generated.fasta ]]; then
    echo "Using existing data/processed/de_novo/generated.fasta"
    return
  fi
  echo "Pulling de novo FASTA from s3://${S3_BUCKET}/${S3_PREFIX}/de_novo/generated.fasta"
  if command -v aws >/dev/null 2>&1; then
    aws s3 cp "s3://${S3_BUCKET}/${S3_PREFIX}/de_novo/generated.fasta" \
      data/processed/de_novo/generated.fasta
  else
    python - <<PY
import os
from pathlib import Path
import boto3
bucket = os.environ.get("AWS_S3_BUCKET", "${S3_BUCKET}")
prefix = os.environ.get("AWS_S3_PREFIX", "${S3_PREFIX}")
key = f"{prefix}/de_novo/generated.fasta"
out = Path("data/processed/de_novo/generated.fasta")
out.parent.mkdir(parents=True, exist_ok=True)
boto3.client("s3", region_name=os.environ.get("AWS_REGION", "us-east-1")).download_file(
    bucket, key, str(out)
)
print(f"Downloaded s3://{bucket}/{key} -> {out}")
PY
  fi
}

clear_feature_outputs() {
  echo "Clearing prior feature outputs (avoid mixing EC2 physics params)"
  rm -f \
    data/processed/fused_features.csv \
    data/processed/denovo_fused_features.csv \
    data/processed/denovo_predictions.csv \
    data/processed/viennarna/features.csv \
    data/processed/viennarna/denovo_features.csv \
    data/processed/nupack/features.csv \
    data/processed/nupack/denovo_features.csv \
    data/processed/batch_ram_log.jsonl
  # Keep trained model only if explicitly requested
  if [[ "${KEEP_MODEL:-0}" != "1" ]]; then
    rm -f data/processed/models/rf_thermoswitch.joblib
  fi
}

run_rfam_train() {
  local limit="$1"
  echo "=== Rfam balanced thermo (limit=${limit}) ==="
  python src/thermo_sim/thermo_batch.py \
    "${COMMON_FLAGS[@]}" \
    --input-mode balanced \
    --limit "$limit" \
    --vienna-csv data/processed/viennarna/features.csv \
    --nupack-csv data/processed/nupack/features.csv \
    --fused-csv data/processed/fused_features.csv

  echo "=== Train Random Forest ==="
  python src/thermo_sim/thermo_classifier.py train \
    --fused-csv data/processed/fused_features.csv \
    --model-path data/processed/models/rf_thermoswitch.joblib
}

run_denovo_predict() {
  local limit="$1"
  echo "=== De novo thermo (limit=${limit}) ==="
  python src/thermo_sim/thermo_batch.py \
    "${COMMON_FLAGS[@]}" \
    --input-mode fasta \
    --input-fasta data/processed/de_novo/generated.fasta \
    --limit "$limit" \
    --vienna-csv data/processed/viennarna/denovo_features.csv \
    --nupack-csv data/processed/nupack/denovo_features.csv \
    --fused-csv data/processed/denovo_fused_features.csv

  echo "=== Predict de novo thermoswitches ==="
  python src/thermo_sim/thermo_classifier.py predict \
    --fused-csv data/processed/denovo_fused_features.csv \
    --model-path data/processed/models/rf_thermoswitch.joblib \
    --predictions-csv data/processed/denovo_predictions.csv
}

preflight() {
  test -f data/processed/balanced/balanced_dataset.csv
  test -f data/processed/balanced/balanced_dataset.fasta
  python - <<'PY'
import RNA  # noqa: F401
from nupack import Model  # noqa: F401
print("ViennaRNA + NUPACK imports OK")
PY
}

echo "Mac M3 thermo local — mode=${MODE} workers=${WORKERS} temps=${TEMPS} Mg=${MAGNESIUM}"
preflight
pull_denovo_fasta
clear_feature_outputs

case "$MODE" in
  smoke)
    # Balanced set is sorted label=0 then label=1; build a 2+2 mini panel for RF.
    PYTHONPATH=src python - <<'PY'
import pandas as pd
from pathlib import Path
from data_engineering.paths import resolve_path
from thermo_sim.thermo_common import load_balanced_dataset, normalize_sequence

df = load_balanced_dataset()
mini = pd.concat([df[df["label"] == 0].head(2), df[df["label"] == 1].head(2)])
out_dir = resolve_path("data/processed/prototype")
out_dir.mkdir(parents=True, exist_ok=True)
csv_path = out_dir / "mac_smoke_panel.csv"
fasta_path = out_dir / "mac_smoke_panel.fasta"
mini.drop(columns=["sequence", "seq_length"], errors="ignore").to_csv(csv_path, index=False)
with fasta_path.open("w") as handle:
    for _, row in mini.iterrows():
        handle.write(
            f">{row['rfam_acc']}|{row['rfam_id']}|{row['rfamseq_acc']}|"
            f"{int(row['seq_start'])}-{int(row['seq_end'])}|label={int(row['label'])}\n"
            f"{normalize_sequence(row['sequence'])}\n"
        )
print(f"Wrote smoke panel {len(mini)} rows -> {csv_path}")
PY
    echo "=== Rfam smoke panel thermo ==="
    python src/thermo_sim/thermo_batch.py \
      "${COMMON_FLAGS[@]}" \
      --input-mode balanced \
      --input-csv data/processed/prototype/mac_smoke_panel.csv \
      --input-fasta data/processed/prototype/mac_smoke_panel.fasta \
      --limit 4 \
      --vienna-csv data/processed/viennarna/features.csv \
      --nupack-csv data/processed/nupack/features.csv \
      --fused-csv data/processed/fused_features.csv
    echo "=== Train Random Forest ==="
    python src/thermo_sim/thermo_classifier.py train \
      --fused-csv data/processed/fused_features.csv \
      --model-path data/processed/models/rf_thermoswitch.joblib
    run_denovo_predict 4
    echo "Smoke complete."
    wc -l data/processed/fused_features.csv data/processed/denovo_fused_features.csv data/processed/denovo_predictions.csv
    ;;
  full)
    run_rfam_train 2396
    run_denovo_predict 10000
    echo "Full Mac thermo pipeline complete."
    wc -l data/processed/fused_features.csv data/processed/denovo_fused_features.csv data/processed/denovo_predictions.csv
    ;;
  *)
    echo "Usage: bash scripts/mac_thermo_local.sh [smoke|full]"
    exit 1
    ;;
esac
