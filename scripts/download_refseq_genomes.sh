#!/usr/bin/env bash
# Download a capped set of complete reference genomes (FASTA + GFF3) for UTR negatives.
# Uses datasets summary → accession list → download genome accession (tractable on laptop).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p logs

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

: "${NCBI_API_KEY:?Set NCBI_API_KEY in .env}"

export PATH="${ROOT}/.tools/bin:${PATH}"
if ! command -v datasets >/dev/null 2>&1; then
  echo "datasets CLI not found in PATH / .tools/bin" >&2
  exit 1
fi

OUT_ROOT="${REFSEQ_GENOME_DIR:-data/raw/refseq_genomes}"
PER_TAXON="${ASSEMBLIES_PER_TAXON:-40}"
MIN_ASSEMBLIES="${MIN_ASSEMBLIES:-50}"
mkdir -p "$OUT_ROOT"

pick_accessions() {
  local taxon="$1"
  local n="$2"
  local jsonl="${OUT_ROOT}/${taxon}/summary.jsonl"
  mkdir -p "${OUT_ROOT}/${taxon}"
  echo "=== Summarizing taxon=${taxon} ===" >&2
  datasets summary genome taxon "${taxon}" \
    --assembly-level complete \
    --reference \
    --annotated \
    --exclude-atypical \
    --as-json-lines \
    --api-key "${NCBI_API_KEY}" \
    > "${jsonl}"
  # Prefer RefSeq GCF_ accessions
  python3 - "${jsonl}" "${n}" <<'PY'
import json, sys
path, n = sys.argv[1], int(sys.argv[2])
accs = []
with open(path) as fh:
    for line in fh:
        line=line.strip()
        if not line: continue
        try:
            obj=json.loads(line)
        except json.JSONDecodeError:
            continue
        # datasets summary schema variants
        acc = None
        if "accession" in obj:
            acc = obj["accession"]
        elif "assembly" in obj and isinstance(obj["assembly"], dict):
            acc = obj["assembly"].get("assembly_accession") or obj["assembly"].get("accession")
        elif "assembly_info" in obj:
            acc = obj.get("accession") or obj["assembly_info"].get("assembly_accession")
        # nested reports
        if acc is None and "reports" in obj:
            continue
        if not acc:
            # try common path
            acc = (
                obj.get("assembly", {})
                .get("assembly_accession")
                if isinstance(obj.get("assembly"), dict)
                else None
            )
        if acc and str(acc).startswith("GCF_"):
            accs.append(str(acc))
        elif acc and str(acc).startswith("GCA_") and len(accs) < n:
            accs.append(str(acc))
# unique preserve order
seen=set(); out=[]
for a in accs:
    if a in seen: continue
    seen.add(a); out.append(a)
    if len(out)>=n: break
print("\n".join(out))
PY
}

download_accessions() {
  local taxon="$1"
  local list_file="$2"
  local dest="${OUT_ROOT}/${taxon}"
  local zip="${dest}/genomes.zip"
  mkdir -p "${dest}"
  local count
  count="$(grep -c . "${list_file}" || true)"
  echo "=== Downloading ${count} assemblies for ${taxon} ==="
  if [[ "${count}" -lt 1 ]]; then
    echo "ERROR: no accessions for ${taxon}" >&2
    exit 1
  fi
  datasets download genome accession \
    --inputfile "${list_file}" \
    --include genome,gff3 \
    --filename "${zip}" \
    --api-key "${NCBI_API_KEY}"
  unzip -o -q "${zip}" -d "${dest}"
  local n
  n="$(find "${dest}" -type d \( -name 'GCF_*' -o -name 'GCA_*' \) | wc -l | tr -d ' ')"
  echo "  assemblies on disk: ${n}"
  echo "${n}" > "${dest}/assembly_count.txt"
}

for TAXON in Pseudomonadota Bacillota; do
  mkdir -p "${OUT_ROOT}/${TAXON}"
  LIST="${OUT_ROOT}/${TAXON}/accessions.txt"
  # Skip re-download if assemblies already present (resume-friendly)
  EXISTING="$(find "${OUT_ROOT}/${TAXON}" -type d \( -name 'GCF_*' -o -name 'GCA_*' \) 2>/dev/null | wc -l | tr -d ' ')"
  if [[ "${EXISTING}" -ge "${PER_TAXON}" ]]; then
    echo "=== Skipping ${TAXON}: already have ${EXISTING} assemblies ==="
    continue
  fi
  pick_accessions "${TAXON}" "${PER_TAXON}" > "${LIST}"
  echo "Selected $(grep -c . "${LIST}" || echo 0) accessions for ${TAXON}"
  head -n 5 "${LIST}" || true
  download_accessions "${TAXON}" "${LIST}"
done

TOTAL="$(find "${OUT_ROOT}" -type d \( -name 'GCF_*' -o -name 'GCA_*' \) | wc -l | tr -d ' ')"
echo "=== Total assemblies: ${TOTAL} (minimum ${MIN_ASSEMBLIES}) ==="
if [[ "${TOTAL}" -lt "${MIN_ASSEMBLIES}" ]]; then
  echo "ERROR: need >= ${MIN_ASSEMBLIES} assemblies; got ${TOTAL}" >&2
  exit 2
fi
echo "OK: genomes ready under ${OUT_ROOT}"
