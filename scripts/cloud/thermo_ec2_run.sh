#!/usr/bin/env bash
# SSH wrapper to run thermo_s3_batch.py on aws-thermo-ec2.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$ROOT"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

: "${EC2_HOST:?Set EC2_HOST in .env}"
: "${EC2_USER:?Set EC2_USER in .env}"
: "${EC2_SSH_KEY:?Set EC2_SSH_KEY in .env}"
: "${EC2_REPO_PATH:=~/Thermoswitches-MLBIO}"

KEY="${EC2_SSH_KEY/#\~/$HOME}"
REMOTE="${EC2_USER}@${EC2_HOST}"

exec ssh -i "$KEY" "$REMOTE" \
  "cd ${EC2_REPO_PATH} && source .venv/bin/activate && python scripts/cloud/thermo_s3_batch.py $*"
