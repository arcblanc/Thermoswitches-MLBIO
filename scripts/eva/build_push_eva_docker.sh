#!/usr/bin/env bash
# Bake and optionally push the EVA CUDA Docker image for RunPod.
#
# Run this ON a temporary Linux x86_64 VM (after SSH from your M3 Mac).
# Do NOT run docker build for EVA CUDA images on Apple Silicon — they need
# pytorch/cuda + flash-attn linux/amd64 toolchains.
#
# Usage (on the VM):
#   bash scripts/eva/build_push_eva_docker.sh           # build only
#   bash scripts/eva/build_push_eva_docker.sh --push    # build + docker push
#
# Env (optional):
#   DOCKERHUB_IMAGE=arcblanc/eva-model:v1
#   EVA_SRC_DIR=~/EVA
#   EVA_GIT_URL=https://github.com/GENTEL-Lab/EVA.git
set -euo pipefail

DOCKERHUB_IMAGE="${DOCKERHUB_IMAGE:-arcblanc/eva-model:v1}"
EVA_SRC_DIR="${EVA_SRC_DIR:-$HOME/EVA}"
EVA_GIT_URL="${EVA_GIT_URL:-https://github.com/GENTEL-Lab/EVA.git}"
DO_PUSH=0

for arg in "$@"; do
  case "${arg}" in
    --push) DO_PUSH=1 ;;
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

# --- Hard-fail on Darwin / non-amd64 (M3 Mac must SSH to a Linux VM) -------
OS_NAME="$(uname -s)"
ARCH_NAME="$(uname -m)"
if [[ "${OS_NAME}" == "Darwin" ]]; then
  cat >&2 <<'EOF'
ERROR: Refusing to bake EVA CUDA Docker images on macOS (e.g. M3).

Your Mac is for local biophysics (ViennaRNA / NUPACK) and SSH only.
CUDA + flash-attn layers must be built on a temporary Linux x86_64 VM:

  1. SSH into a cheap Linux amd64 cloud VM with Docker installed
  2. Clone Thermoswitches-MLBIO (or copy this script)
  3. On the VM:  bash scripts/eva/build_push_eva_docker.sh --push

See cluster/EVA_RUNPOD.md → "Bake EVA image (Docker Hub)".
EOF
  exit 1
fi

if [[ "${ARCH_NAME}" != "x86_64" && "${ARCH_NAME}" != "amd64" ]]; then
  echo "ERROR: Expected Linux x86_64/amd64 host; got arch=${ARCH_NAME}." >&2
  echo "Bake EVA CUDA images only on linux/amd64." >&2
  exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: docker not found on PATH. Install Docker Engine on this VM." >&2
  exit 1
fi

# Expand ~
EVA_SRC_DIR="${EVA_SRC_DIR/#\~/$HOME}"

echo "=== EVA Docker bake ==="
echo "  host:   ${OS_NAME} ${ARCH_NAME}"
echo "  image:  ${DOCKERHUB_IMAGE}"
echo "  source: ${EVA_SRC_DIR}"
echo "  push:   ${DO_PUSH}"

if [[ ! -d "${EVA_SRC_DIR}/.git" ]]; then
  echo "=== Cloning ${EVA_GIT_URL} → ${EVA_SRC_DIR} ==="
  mkdir -p "$(dirname "${EVA_SRC_DIR}")"
  git clone "${EVA_GIT_URL}" "${EVA_SRC_DIR}"
else
  echo "=== Updating existing clone ==="
  git -C "${EVA_SRC_DIR}" fetch --tags --force
  git -C "${EVA_SRC_DIR}" pull --ff-only || true
fi

if [[ ! -f "${EVA_SRC_DIR}/docker/Dockerfile" ]]; then
  echo "ERROR: missing ${EVA_SRC_DIR}/docker/Dockerfile" >&2
  exit 1
fi

# Upstream FROM tag …-runtime-ubuntu22.04 was removed from Hub.
# Prefer a local override Dockerfile so we do not rewrite the clone permanently.
# Also: upstream .dockerignore excludes docker/, which breaks COPY docker/requirements.txt.
DOCKERFILE="${EVA_SRC_DIR}/docker/Dockerfile"
OVERRIDE_DF="${EVA_SRC_DIR}/docker/Dockerfile.arcblanc"

# devel base: grouped_gemm / megablocks still need nvcc (bake VM has no GPU driver).
BASE_IMAGE="${EVA_BASE_IMAGE:-pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel}"
# Official v2.6.3 has no cu124/torch2.5 wheel (404). Closest verified wheel for
# PyTorch 2.5 + cxx11abiFALSE + cp311 is flash-attn 2.8.3.post1 (HTTP 200).
FLASH_ATTN_WHEEL="${FLASH_ATTN_WHEEL:-https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post1/flash_attn-2.8.3.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl}"
CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"

cat > "${OVERRIDE_DF}" <<EOF
# Arcblanc bake override — derived from EVA docker/Dockerfile
# Prebuilt flash-attn wheel; CUDA extension deps compiled with driver shim.
FROM ${BASE_IMAGE}

RUN apt-get update && apt-get install -y --no-install-recommends git ninja-build \\
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir "${FLASH_ATTN_WHEEL}"

COPY docker/requirements.txt /tmp/requirements.txt
# Drop packages already in the base image / installed via wheel, plus python-apt
# (Ubuntu .deb pin) and CUDA extensions that need a driver-shimmed install.
RUN grep -vE '^(flash[_-]attn|torch==|torchvision==|triton==|nvidia-|python-apt|grouped_gemm|megablocks|stanford-stk)' \\
      /tmp/requirements.txt > /tmp/requirements.nofa.txt \\
    && pip install --no-cache-dir -r /tmp/requirements.nofa.txt \\
    && rm -f /tmp/requirements.txt /tmp/requirements.nofa.txt

# Bake VM has no NVIDIA driver; patch is_available so setup.py can build for sm_80.
ENV TORCH_CUDA_ARCH_LIST="${CUDA_ARCH_LIST}"
ENV FORCE_CUDA=1
ENV MAX_JOBS=2
RUN python - <<'PY'
import os, subprocess, sys
import torch
torch.cuda.is_available = lambda: True
pkgs = ["grouped_gemm==0.1.6", "stanford-stk==0.7.1", "megablocks==0.7.0"]
subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "--no-cache-dir", "--no-build-isolation", *pkgs],
    env={**os.environ, "TORCH_CUDA_ARCH_LIST": os.environ.get("TORCH_CUDA_ARCH_LIST", "8.0"), "FORCE_CUDA": "1"},
)
PY

COPY eva/ /eva/eva/
COPY tools/ /eva/tools/
COPY config/ /eva/config/
COPY data/test_data/ /eva/data/test_data/
COPY pyproject.toml README.md LICENSE CHANGELOG.md /eva/
RUN pip install --no-cache-dir --no-deps -e /eva

RUN mkdir -p /eva/data/output/scores

WORKDIR /eva
ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh"]
CMD ["sleep", "infinity"]
EOF
DOCKERFILE="${OVERRIDE_DF}"
echo "=== Using base image ${BASE_IMAGE} ==="
echo "=== flash-attn wheel: ${FLASH_ATTN_WHEEL} ==="
echo "=== CUDA arch for remaining extensions: ${CUDA_ARCH_LIST} ==="

DOCKERIGNORE_BAK=""
if [[ -f "${EVA_SRC_DIR}/.dockerignore" ]] && grep -qE '^docker/?$' "${EVA_SRC_DIR}/.dockerignore"; then
  DOCKERIGNORE_BAK="${EVA_SRC_DIR}/.dockerignore.arcblanc.bak"
  cp "${EVA_SRC_DIR}/.dockerignore" "${DOCKERIGNORE_BAK}"
  # Un-ignore docker/requirements.txt for the COPY in Dockerfile
  {
    cat "${DOCKERIGNORE_BAK}"
    echo '!docker/'
    echo '!docker/requirements.txt'
  } > "${EVA_SRC_DIR}/.dockerignore"
  echo "=== Patched .dockerignore to allow docker/requirements.txt ==="
fi

cleanup_dockerignore() {
  if [[ -n "${DOCKERIGNORE_BAK}" && -f "${DOCKERIGNORE_BAK}" ]]; then
    mv "${DOCKERIGNORE_BAK}" "${EVA_SRC_DIR}/.dockerignore"
  fi
}
trap cleanup_dockerignore EXIT

echo "=== docker build (prebuilt flash-attn wheel; should be minutes, not hours) ==="
echo "  dockerfile: ${DOCKERFILE}"
docker build \
  -f "${DOCKERFILE}" \
  -t "${DOCKERHUB_IMAGE}" \
  "${EVA_SRC_DIR}"

cleanup_dockerignore
trap - EXIT

echo "=== Verifying image has eva-generate ==="
docker run --rm --entrypoint bash "${DOCKERHUB_IMAGE}" -lc 'command -v eva-generate && eva-generate --help | head -20'

if [[ "${DO_PUSH}" -eq 1 ]]; then
  echo "=== docker push ${DOCKERHUB_IMAGE} ==="
  if ! docker info 2>/dev/null | grep -qi 'Username'; then
    echo "Not logged in to Docker Hub (or credentials unknown)."
    echo "Run: docker login"
    docker login
  fi
  docker push "${DOCKERHUB_IMAGE}"
  echo "Pushed ${DOCKERHUB_IMAGE}"
  echo "Next: create a RunPod template with Container Image = ${DOCKERHUB_IMAGE}"
else
  echo "Build complete (not pushed). Re-run with --push after: docker login"
fi
