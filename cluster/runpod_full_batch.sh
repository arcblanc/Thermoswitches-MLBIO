#!/usr/bin/env bash
# Paste this entire block into the RunPod SSH web terminal (Thermopod).
# Requires: AWS keys, HF_TOKEN, RUNPOD_API_KEY in environment or .env on pod.
set -euo pipefail

export STORAGE_TARGET=runpod
export GENERNA_NUM_SAMPLES=10000
export GENERNA_BATCH_SIZE=50
export RUNPOD_AUTO_TERMINATE=true
export RUNPOD_POD_ID=x0ggh3d7lmi9yn

REPO="${REPO:-$HOME/Thermoswitches-MLBIO}"
if [[ ! -d "$REPO/.git" ]]; then
  git clone https://github.com/arcblanc/Thermoswitches-MLBIO.git "$REPO"
fi
cd "$REPO"
git pull origin main

python3 -m venv .venv
source .venv/bin/activate
pip install -q -r requirements-llm.txt -r requirements-aws.txt

# Create .env if missing (paste your secrets here before running):
if [[ ! -f .env ]]; then
  cat > .env <<'ENVEOF'
STORAGE_TARGET=runpod
AWS_S3_BUCKET=thermo-s3-bucket
AWS_REGION=us-east-1
AWS_S3_PREFIX=llm-batch/v1
AWS_ACCESS_KEY_ID=PASTE_YOUR_KEY
AWS_SECRET_ACCESS_KEY=PASTE_YOUR_SECRET
HF_TOKEN=PASTE_YOUR_HF_TOKEN
RUNPOD_POD_ID=x0ggh3d7lmi9yn
RUNPOD_API_KEY=PASTE_YOUR_RUNPOD_API_KEY
RUNPOD_AUTO_TERMINATE=true
GENERNA_NUM_SAMPLES=10000
GENERNA_BATCH_SIZE=50
ENVEOF
  echo "Created .env template — edit secrets then re-run this script."
  exit 1
fi

bash cluster/runpod_thermopod.sh --yes
