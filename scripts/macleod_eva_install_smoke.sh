#!/usr/bin/env bash
# Install EVA (GENTEL-Lab) into the active conda env and run a tiny GPU smoke
# generate on Macleod gpu02 MIG. Run INSIDE an srun GPU job with torch_mig active.
#
# Usage:
#   module load miniconda3 cuda/12.4.0
#   conda activate torch_mig
#   export CUDA_VISIBLE_DEVICES=0
#   bash scripts/macleod_eva_install_smoke.sh
#   bash scripts/macleod_eva_install_smoke.sh --skip-download   # reuse checkpoint
#   bash scripts/macleod_eva_install_smoke.sh --repo-smoke      # also run eva_generate --smoke
#   bash scripts/macleod_eva_install_smoke.sh --skip-birna      # with --repo-smoke: generate only
set -euo pipefail

EVA_SRC_DIR="${EVA_SRC_DIR:-$HOME/EVA}"
EVA_CHECKPOINT_DIR="${EVA_CHECKPOINT_DIR:-$HOME/eva_checkpoint}"
EVA_SMOKE_NUM_SEQS="${EVA_SMOKE_NUM_SEQS:-4}"
EVA_SMOKE_OUT="${EVA_SMOKE_OUT:-$HOME/eva_smoke_out}"
EVA_GIT_URL="${EVA_GIT_URL:-https://github.com/GENTEL-Lab/EVA.git}"
SKIP_DOWNLOAD=0
REPO_SMOKE=0
SKIP_BIRNA=1

for arg in "$@"; do
  case "${arg}" in
    --skip-download) SKIP_DOWNLOAD=1 ;;
    --repo-smoke) REPO_SMOKE=1 ;;
    --with-birna) SKIP_BIRNA=0 ;;
    --skip-birna) SKIP_BIRNA=1 ;;
    -h|--help)
      sed -n '2,20p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown arg: ${arg}" >&2
      exit 1
      ;;
  esac
done

echo "=== Macleod EVA install/smoke ==="
echo "  EVA_SRC_DIR=${EVA_SRC_DIR}"
echo "  EVA_CHECKPOINT_DIR=${EVA_CHECKPOINT_DIR}"
echo "  num_seqs=${EVA_SMOKE_NUM_SEQS}"

# --- 1. CUDA -----------------------------------------------------------------
python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable — run inside srun GPU job with torch_mig activated")
print(f"torch {torch.__version__} | {torch.cuda.get_device_name(0)}")
print(f"VRAM_GiB {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f}")
PY

# --- 2. Clone + install EVA --------------------------------------------------
if [[ ! -d "${EVA_SRC_DIR}/.git" ]]; then
  echo "=== Cloning ${EVA_GIT_URL} → ${EVA_SRC_DIR} ==="
  mkdir -p "$(dirname "${EVA_SRC_DIR}")"
  git clone "${EVA_GIT_URL}" "${EVA_SRC_DIR}"
else
  echo "=== Updating ${EVA_SRC_DIR} ==="
  git -C "${EVA_SRC_DIR}" pull --ff-only || true
fi

echo "=== pip install -e EVA ==="
python -m pip install -U pip setuptools wheel
python -m pip install -e "${EVA_SRC_DIR}"

if ! python -c "import flash_attn" 2>/dev/null; then
  echo "=== Attempting flash-attn (may take a long time) ==="
  python -m pip install flash-attn --no-build-isolation || {
    echo "WARNING: flash-attn install failed. MoE inference may still fail until this works."
  }
fi

if ! command -v eva-generate >/dev/null 2>&1; then
  echo "ERROR: eva-generate not on PATH after install" >&2
  exit 1
fi
eva-generate --help | head -n 5

# --- 3. Checkpoint -----------------------------------------------------------
if [[ "${SKIP_DOWNLOAD}" -eq 1 ]]; then
  echo "=== Skipping checkpoint download (--skip-download) ==="
else
  echo "=== HuggingFace download GENTEL-Lab/EVA → ${EVA_CHECKPOINT_DIR} ==="
  python -m pip install -q "huggingface_hub[cli]"
  mkdir -p "${EVA_CHECKPOINT_DIR}"
  if [[ -n "${HF_TOKEN:-}" ]]; then
    huggingface-cli download GENTEL-Lab/EVA --local-dir "${EVA_CHECKPOINT_DIR}" --token "${HF_TOKEN}"
  else
    huggingface-cli download GENTEL-Lab/EVA --local-dir "${EVA_CHECKPOINT_DIR}"
  fi
fi

if [[ ! -d "${EVA_CHECKPOINT_DIR}" ]]; then
  echo "ERROR: checkpoint dir missing: ${EVA_CHECKPOINT_DIR}" >&2
  exit 1
fi

# --- 4. Tiny official CLI smoke ---------------------------------------------
mkdir -p "${EVA_SMOKE_OUT}"
OUT_FA="${EVA_SMOKE_OUT}/smoke_ecoli.fa"
echo "=== eva-generate smoke (${EVA_SMOKE_NUM_SEQS} seqs, taxid 562) → ${OUT_FA} ==="
eva-generate \
  --checkpoint "${EVA_CHECKPOINT_DIR}" \
  --format clm \
  --rna_type mRNA \
  --taxid 562 \
  --num_seqs "${EVA_SMOKE_NUM_SEQS}" \
  --output "${OUT_FA}"

echo "=== Smoke FASTA preview ==="
wc -l "${OUT_FA}" || true
head -n 20 "${OUT_FA}" || true

# --- 5. Optional repo orchestrator ------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
if [[ ! -f "${REPO_ROOT}/src/de_novo_hallucinations/eva_generate.py" ]]; then
  REPO_ROOT="${THERMO_REPO:-$HOME/Thermoswitches-MLBIO}"
fi

if [[ "${REPO_SMOKE}" -eq 1 ]]; then
  if [[ ! -f "${REPO_ROOT}/src/de_novo_hallucinations/eva_generate.py" ]]; then
    echo "WARNING: Thermoswitches-MLBIO not found at ${REPO_ROOT}; skip --repo-smoke"
  else
    echo "=== Repo eva_generate --smoke (STORAGE_TARGET=local) ==="
    cd "${REPO_ROOT}"
    export STORAGE_TARGET=local
    export EVA_RNA_TYPE=mRNA
    export EVA_CHECKPOINT_DIR
    export EVA_NUM_SAMPLES="${EVA_NUM_SAMPLES:-16}"
    export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
    python -m pip install -q -r requirements-llm.txt || true
    if [[ "${SKIP_BIRNA}" -eq 1 ]]; then
      python src/de_novo_hallucinations/eva_generate.py --smoke
    else
      python scripts/eva_cloud_batch.py --smoke
    fi
  fi
fi

echo ""
echo "=== Macleod EVA tiny smoke finished ==="
echo "  FASTA: ${OUT_FA}"
echo "  Next: bash scripts/macleod_eva_install_smoke.sh --skip-download --repo-smoke"
