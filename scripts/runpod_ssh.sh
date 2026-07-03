#!/usr/bin/env bash
# Open SSH session to RunPod Thermopod (SSH only — no SCP/SFTP on RunPod).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

: "${RUNPOD_SSH_USER:?Set RUNPOD_SSH_USER in .env}"
: "${RUNPOD_SSH_HOST:=ssh.runpod.io}"
: "${RUNPOD_SSH_KEY:?Set RUNPOD_SSH_KEY in .env}"

exec ssh -i "${RUNPOD_SSH_KEY/#\~/$HOME}" "${RUNPOD_SSH_USER}@${RUNPOD_SSH_HOST}"
