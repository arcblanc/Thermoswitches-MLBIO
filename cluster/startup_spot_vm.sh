#!/usr/bin/env bash
# GCE startup script — clone repo, install LLM deps, run cloud batch, self-delete on success.
set -euo pipefail

LOG=/var/log/thermoswitches-llm-startup.log
exec > >(tee -a "$LOG") 2>&1

echo "=== Thermoswitches LLM startup $(date -Is) ==="

apt-get update
apt-get install -y git python3-venv python3-pip curl

METADATA_URL="http://metadata.google.internal/computeMetadata/v1/instance/attributes"
metadata() {
  curl -sf -H "Metadata-Flavor: Google" "${METADATA_URL}/$1" || true
}

GCS_BUCKET="$(metadata gcs-bucket)"
HF_TOKEN="$(metadata hf-token)"
GENERNA_NUM_SAMPLES="$(metadata generna-num-samples)"
REPO_URL="$(metadata repo-url)"

: "${REPO_URL:=https://github.com/arcblanc/Thermoswitches-MLBIO.git}"
: "${GENERNA_NUM_SAMPLES:=10000}"

INSTALL_DIR=/opt/thermoswitches-mlbio
rm -rf "$INSTALL_DIR"
git clone "$REPO_URL" "$INSTALL_DIR"
cd "$INSTALL_DIR"

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-llm.txt

cat > .env <<EOF
STORAGE_TARGET=gcs
GCS_BUCKET=${GCS_BUCKET}
GCS_PREFIX=llm-batch/v1
VM_AUTO_SHUTDOWN=true
GENERNA_BATCH_SIZE=50
GENERNA_NUM_SAMPLES=${GENERNA_NUM_SAMPLES}
HF_TOKEN=${HF_TOKEN}
GENERNA_CACHE_DIR=models/genererna
BIRNA_TOKENIZER_ID=buetnlpbio/birna-tokenizer
BIRNA_MODEL_ID=buetnlpbio/birna-bert
EMBEDDING_OUTPUT_DIR=data/processed/validation_embedding
DE_NOVO_OUTPUT_DIR=data/processed/de_novo
EOF

bash scripts/llm_cloud_run.sh

echo "=== Startup complete $(date -Is) ==="
