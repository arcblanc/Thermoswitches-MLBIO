#!/usr/bin/env bash
# Request GPU quotas for thermoswitchmlbio via gcloud (requires beta component).
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-thermoswitchmlbio}"
REGION="${REGION:-us-central1}"
CONTACT="${QUOTA_CONTACT_EMAIL:-$(gcloud config get-value account)}"

echo "Project: ${PROJECT_ID}"
echo "Contact: ${CONTACT}"
echo "Region:  ${REGION}"

gcloud components install alpha --quiet 2>/dev/null || true

request_quota() {
  local quota_id="$1"
  local value="$2"
  shift 2
  local extra=()
  while [[ $# -ge 2 ]]; do
    extra+=(--dimensions="${1}=${2}")
    shift 2
  done
  echo "Requesting ${quota_id} -> ${value}"
  gcloud alpha quotas preferences create \
    --project="${PROJECT_ID}" \
    --service=compute.googleapis.com \
    --quota-id="${quota_id}" \
    --preferred-value="${value}" \
    --email="${CONTACT}" \
    --justification="Thermoswitches-MLBIO GenerRNA + BiRNA-BERT de novo RNA generation batch job." \
    "${extra[@]}" \
    2>&1 || echo "  (request may already exist or need console follow-up)"
}

# Global GPU cap (required before any GPU VM on new projects)
request_quota "GPUS-ALL-REGIONS-per-project" 1

# A100 80GB for a2-ultragpu-1g Spot jobs in us-central1
request_quota "NVIDIA-A100-80GB-GPUS-per-project-region" 1 region "${REGION}"

echo ""
echo "Track requests: https://console.cloud.google.com/iam-admin/quotas?project=${PROJECT_ID}"
