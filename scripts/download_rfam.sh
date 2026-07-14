#!/usr/bin/env bash
# Download Rfam 14.9 FASTA reference for novelty search.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RFAM_DIR="${ROOT}/data/reference/rfam/14.9"
URL="https://ftp.ebi.ac.uk/pub/databases/Rfam/14.9/fasta_files/Rfam.fa.gz"

mkdir -p "${RFAM_DIR}"
cd "${RFAM_DIR}"

if [[ -f Rfam.fa ]]; then
  echo "Rfam.fa already present at ${RFAM_DIR}/Rfam.fa"
  exit 0
fi

echo "Downloading ${URL} ..."
curl -L -o Rfam.fa.gz "${URL}"
gunzip -k Rfam.fa.gz
echo "Ready: ${RFAM_DIR}/Rfam.fa ($(wc -c < Rfam.fa | awk '{printf "%.1f MB", $1/1e6}'))"
