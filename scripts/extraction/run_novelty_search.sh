#!/usr/bin/env bash
# End-to-end Rfam novelty search: extract queries, BLAST, nhmmer, report.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

RFAM_FA="data/reference/rfam/14.9/Rfam.fa"
RFAM_DB="data/reference/rfam/14.9/rfam"
QUERY_FASTA="data/processed/novelty/candidates_99.fasta"
BLAST_TSV="data/processed/novelty/blastn_hits.tsv"
NHMMER_TBL="data/processed/novelty/nhmmer_hits.tbl"
NHMMER_OUT="data/processed/novelty/nhmmer_hits.out"
EVALUE="0.1"

PYTHON="${PYTHON:-python3}"
if [[ -x "${ROOT}/.venv/bin/python" ]]; then
  PYTHON="${ROOT}/.venv/bin/python"
fi

export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

mkdir -p data/processed/novelty

echo "=== Extract 99 candidate sequences ==="
"${PYTHON}" -m novelty_eval.extract_candidates

if [[ ! -f "${RFAM_FA}" ]]; then
  echo "Rfam.fa missing — run: bash scripts/extraction/download_rfam.sh"
  exit 1
fi

if [[ "${SKIP_SEARCH:-0}" != "1" ]]; then
  for cmd in makeblastdb blastn nhmmer; do
    if ! command -v "${cmd}" >/dev/null 2>&1; then
      echo "${cmd} not found. Install: conda env update -f environment.yml"
      exit 1
    fi
  done

  echo "=== Build BLAST database (skip if present) ==="
  if [[ ! -f "${RFAM_DB}.nhr" ]]; then
    makeblastdb -in "${RFAM_FA}" -dbtype nucl -out "${RFAM_DB}"
  fi

  echo "=== blastn (-task blastn, E-value ${EVALUE}) ==="
  blastn \
    -task blastn \
    -query "${QUERY_FASTA}" \
    -db "${RFAM_DB}" \
    -evalue "${EVALUE}" \
    -max_target_seqs 5 \
    -num_threads "${NOVELTY_THREADS:-4}" \
    -outfmt "6 qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore" \
    -out "${BLAST_TSV}"

  echo "=== nhmmer (-E ${EVALUE}, --rna) ==="
  nhmmer \
    --rna \
    -E "${EVALUE}" \
    --tblout "${NHMMER_TBL}" \
    -o "${NHMMER_OUT}" \
    --cpu "${NOVELTY_THREADS:-4}" \
    "${QUERY_FASTA}" \
    "${RFAM_FA}"
else
  echo "SKIP_SEARCH=1 — reusing existing BLAST/nhmmer outputs"
fi

echo "=== Novelty report ==="
"${PYTHON}" -m novelty_eval.novelty_report

echo "=== Done ==="
