#!/usr/bin/env bash
# Novelty search for all sequences in an EVA/pilot gated FASTA (blastn + nhmmer vs Rfam 14.9).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

SOURCE_FASTA="${SOURCE_FASTA:-data/processed/eva_pilot/generated.fasta}"
CANDIDATES_CSV="${CANDIDATES_CSV:-data/processed/eva_pilot/candidate_ids.csv}"
QUERY_FASTA="${QUERY_FASTA:-data/processed/novelty/eva_pilot_queries.fasta}"
RFAM_FA="data/reference/rfam/14.9/Rfam.fa"
RFAM_DB="data/reference/rfam/14.9/rfam"
BLAST_TSV="${BLAST_TSV:-data/processed/novelty/blastn_hits.tsv}"
NHMMER_TBL="${NHMMER_TBL:-data/processed/novelty/nhmmer_hits.tbl}"
NHMMER_OUT="${NHMMER_OUT:-data/processed/novelty/nhmmer_hits.out}"
EVALUE="${EVALUE:-0.1}"

PYTHON="${PYTHON:-python3}"
if [[ -x "${ROOT}/.venv/bin/python" ]]; then
  PYTHON="${ROOT}/.venv/bin/python"
fi

export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
mkdir -p data/processed/novelty data/processed/eva_pilot

echo "=== Extract all sequences from ${SOURCE_FASTA} ==="
"${PYTHON}" -m novelty_eval.extract_candidates \
  --from-source-fasta \
  --source-fasta "${SOURCE_FASTA}" \
  --candidates-csv "${CANDIDATES_CSV}" \
  --output-fasta "${QUERY_FASTA}"

if [[ ! -f "${RFAM_FA}" ]]; then
  echo "Rfam.fa missing — run: bash scripts/download_rfam.sh"
  exit 1
fi

if [[ "${SKIP_SEARCH:-0}" != "1" ]]; then
  for cmd in makeblastdb blastn nhmmer; do
    if ! command -v "${cmd}" >/dev/null 2>&1; then
      echo "${cmd} not found. Install: conda env update -f environment.yml"
      exit 1
    fi
  done

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
fi

echo "=== Novelty report ==="
"${PYTHON}" -m novelty_eval.novelty_report \
  --candidates-csv "${CANDIDATES_CSV}" \
  --query-fasta "${QUERY_FASTA}" \
  --blast-tsv "${BLAST_TSV}" \
  --nhmmer-tbl "${NHMMER_TBL}" \
  --report-csv data/processed/novelty/eva_pilot_novelty_report.csv \
  --summary-json data/processed/novelty/eva_pilot_novelty_summary.json \
  --by-category-csv data/processed/novelty/eva_pilot_novelty_by_category.csv

echo "=== Done ==="
