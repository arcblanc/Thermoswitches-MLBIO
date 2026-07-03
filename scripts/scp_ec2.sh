#!/usr/bin/env bash
# SCP helpers for aws-thermo-ec2 (standard SSH — SCP supported).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
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
REMOTE_REPO="${EC2_REPO_PATH}"

cmd="${1:-help}"

case "$cmd" in
  push-fasta)
    mkdir -p data/processed/de_novo
    scp -i "$KEY" data/processed/de_novo/generated.fasta \
      "${REMOTE}:${REMOTE_REPO}/data/processed/de_novo/"
    ;;
  push-model)
    scp -i "$KEY" -r data/processed/models/ \
      "${REMOTE}:${REMOTE_REPO}/data/processed/"
    ;;
  pull-fused)
    mkdir -p data/processed
    scp -i "$KEY" \
      "${REMOTE}:${REMOTE_REPO}/data/processed/fused_features.csv" \
      data/processed/
    ;;
  pull-denovo-fused)
    mkdir -p data/processed
    scp -i "$KEY" \
      "${REMOTE}:${REMOTE_REPO}/data/processed/denovo_fused_features.csv" \
      data/processed/
    ;;
  pull-predictions)
    mkdir -p data/processed
    scp -i "$KEY" \
      "${REMOTE}:${REMOTE_REPO}/data/processed/denovo_predictions.csv" \
      data/processed/
    ;;
  pull-all-results)
    mkdir -p data/processed data/processed/models
    scp -i "$KEY" \
      "${REMOTE}:${REMOTE_REPO}/data/processed/fused_features.csv" \
      data/processed/ 2>/dev/null || true
    scp -i "$KEY" \
      "${REMOTE}:${REMOTE_REPO}/data/processed/denovo_fused_features.csv" \
      data/processed/ 2>/dev/null || true
    scp -i "$KEY" \
      "${REMOTE}:${REMOTE_REPO}/data/processed/denovo_predictions.csv" \
      data/processed/ 2>/dev/null || true
    scp -i "$KEY" -r \
      "${REMOTE}:${REMOTE_REPO}/data/processed/models/" \
      data/processed/ 2>/dev/null || true
    ;;
  help|*)
    cat <<EOF
Usage: bash scripts/scp_ec2.sh <command>

Commands:
  push-fasta         SCP generated.fasta to EC2
  push-model         SCP trained RF model to EC2
  pull-fused         Pull training fused_features.csv from EC2
  pull-denovo-fused  Pull de novo fused_features.csv from EC2
  pull-predictions   Pull denovo_predictions.csv from EC2
  pull-all-results   Pull all result artifacts from EC2
EOF
    ;;
esac
