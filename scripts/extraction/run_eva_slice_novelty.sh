#!/usr/bin/env bash
# Slice-scoped novelty (blastn + nhmmer) — does not clobber eva_pilot_* paths.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

SLICE_ID="${SLICE_ID:?Set SLICE_ID (e.g. slice_001)}"
SOURCE_FASTA="${SOURCE_FASTA:?Set SOURCE_FASTA to the slice FASTA}"
OUT_DIR="${OUT_DIR:-data/processed/eva_stream/novelty}"
CANDIDATES_CSV="${CANDIDATES_CSV:-${OUT_DIR}/${SLICE_ID}_candidates.csv}"
QUERY_FASTA="${QUERY_FASTA:-${OUT_DIR}/${SLICE_ID}_queries.fasta}"
BLAST_TSV="${BLAST_TSV:-${OUT_DIR}/${SLICE_ID}_blastn_hits.tsv}"
NHMMER_TBL="${NHMMER_TBL:-${OUT_DIR}/${SLICE_ID}_nhmmer_hits.tbl}"
NHMMER_OUT="${NHMMER_OUT:-${OUT_DIR}/${SLICE_ID}_nhmmer_hits.out}"
REPORT_CSV="${REPORT_CSV:-${OUT_DIR}/${SLICE_ID}_novelty_report.csv}"
SUMMARY_JSON="${SUMMARY_JSON:-${OUT_DIR}/${SLICE_ID}_novelty_summary.json}"
BY_CATEGORY_CSV="${BY_CATEGORY_CSV:-${OUT_DIR}/${SLICE_ID}_novelty_by_category.csv}"
RFAM_FA="data/reference/rfam/14.9/Rfam.fa"
RFAM_DB="data/reference/rfam/14.9/rfam"
EVALUE="${EVALUE:-0.1}"

PYTHON="${PYTHON:-python3}"
if [[ -x "${ROOT}/.venv/bin/python" ]]; then
  PYTHON="${ROOT}/.venv/bin/python"
fi

export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
mkdir -p "${OUT_DIR}"

echo "=== Extract slice ${SLICE_ID} from ${SOURCE_FASTA} ==="
"${PYTHON}" -m novelty_eval.extract_candidates \
  --from-source-fasta \
  --source-fasta "${SOURCE_FASTA}" \
  --candidates-csv "${CANDIDATES_CSV}" \
  --output-fasta "${QUERY_FASTA}"

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

echo "=== Novelty report (${SLICE_ID}) ==="
"${PYTHON}" -m novelty_eval.novelty_report \
  --candidates-csv "${CANDIDATES_CSV}" \
  --query-fasta "${QUERY_FASTA}" \
  --blast-tsv "${BLAST_TSV}" \
  --nhmmer-tbl "${NHMMER_TBL}" \
  --report-csv "${REPORT_CSV}" \
  --summary-json "${SUMMARY_JSON}" \
  --by-category-csv "${BY_CATEGORY_CSV}"

echo "=== Done ${SLICE_ID} ==="
echo "BLAST_TSV=${BLAST_TSV}"
echo "NHMMER_TBL=${NHMMER_TBL}"
