#!/usr/bin/env bash
# Start (or wait for) aws-thermo-ec2, then bake+push EVA Docker image on that Linux host.
#
# From your M3 (orchestrates only — never builds CUDA EVA locally):
#   bash scripts/start_eva_bake_vm.sh
#
# Requirements:
#   - IAM can ec2:DescribeInstances / StartInstances / DescribeInstanceStatus
#     OR the instance is already running and EC2_HOST is reachable
#   - For push: DOCKERHUB_USERNAME + DOCKERHUB_TOKEN in env, or docker login on the VM
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

: "${EC2_USER:=ubuntu}"
: "${EC2_INSTANCE_NAME:=aws-thermo-ec2}"
: "${DOCKERHUB_IMAGE:=arcblanc/eva-model:v1}"
: "${EC2_REPO_PATH:=~/Thermoswitches-MLBIO}"

REGION="${EC2_REGION:-${AWS_REGION:-eu-west-2}}"
KEY_RAW="${EC2_SSH_KEY:-$HOME/Downloads/Thermo-bio-key.pem}"
KEY="${KEY_RAW/#\~/$HOME}"
if [[ ! -f "$KEY" ]]; then
  echo "ERROR: SSH key not found at $KEY" >&2
  echo "Set EC2_SSH_KEY in .env" >&2
  exit 1
fi
chmod 400 "$KEY" 2>/dev/null || true

PYTHON="${ROOT}/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="$(command -v python3)"
fi

HOST_IP=""
STARTED_VIA_API=0

echo "=== Resolving / starting EC2 (${EC2_INSTANCE_NAME}) in ${REGION} ==="
set +e
HOST_IP="$("$PYTHON" - "$ROOT" "$REGION" "$EC2_INSTANCE_NAME" <<'PY'
from pathlib import Path
import sys
from dotenv import dotenv_values
import boto3

root, region, name = sys.argv[1], sys.argv[2], sys.argv[3]
env = dotenv_values(Path(root) / ".env")
iid = (env.get("EC2_INSTANCE_ID") or "").strip() or None

access = env.get("AWS_ACCESS_KEY_ID")
secret = env.get("AWS_SECRET_ACCESS_KEY")
kwargs = {"region_name": region}
if access and secret:
    kwargs["aws_access_key_id"] = access
    kwargs["aws_secret_access_key"] = secret
ec2 = boto3.client("ec2", **kwargs)

def find():
    rows = []
    if iid:
        try:
            d = ec2.describe_instances(InstanceIds=[iid])
            for r in d.get("Reservations", []):
                rows.extend(r.get("Instances", []))
        except Exception as e:
            print(f"WARN: EC2_INSTANCE_ID lookup failed: {e}", file=sys.stderr)
    filt = [
        {"Name": "tag:Name", "Values": [name]},
        {"Name": "instance-state-name", "Values": ["pending", "running", "stopping", "stopped"]},
    ]
    d = ec2.describe_instances(Filters=filt)
    for r in d.get("Reservations", []):
        rows.extend(r.get("Instances", []))
    seen, out = set(), []
    for i in rows:
        if i["InstanceId"] in seen:
            continue
        seen.add(i["InstanceId"])
        out.append(i)
    return out

try:
    insts = find()
except Exception as e:
    print(f"ERROR: cannot describe EC2 instances ({e})", file=sys.stderr)
    print("IAM needs ec2:DescribeInstances (+ StartInstances).", file=sys.stderr)
    print("Or start the instance in the AWS console and set EC2_HOST.", file=sys.stderr)
    sys.exit(2)

if not insts:
    print("ERROR: no instance found for name/id", file=sys.stderr)
    sys.exit(2)

inst = insts[0]
iid = inst["InstanceId"]
state = inst["State"]["Name"]
print(
    f"instance={iid} state={state} type={inst.get('InstanceType')} arch={inst.get('Architecture')}",
    file=sys.stderr,
)

if state == "stopping":
    print("Waiting for stop to finish...", file=sys.stderr)
    ec2.get_waiter("instance_stopped").wait(InstanceIds=[iid])
    state = "stopped"

if state == "stopped":
    print(f"Starting {iid}...", file=sys.stderr)
    ec2.start_instances(InstanceIds=[iid])
    state = "pending"

if state in ("pending", "running"):
    print("Waiting until running + status OK...", file=sys.stderr)
    ec2.get_waiter("instance_running").wait(InstanceIds=[iid])
    ec2.get_waiter("instance_status_ok").wait(InstanceIds=[iid])

d = ec2.describe_instances(InstanceIds=[iid])
ip = d["Reservations"][0]["Instances"][0].get("PublicIpAddress")
if not ip:
    print("ERROR: instance has no PublicIpAddress (check subnet/EIP)", file=sys.stderr)
    sys.exit(3)
print(ip)
PY
)"
api_rc=$?
set -e

if [[ "$api_rc" -eq 0 && -n "$HOST_IP" ]]; then
  STARTED_VIA_API=1
elif [[ -n "${EC2_HOST:-}" ]]; then
  echo "API start unavailable — falling back to EC2_HOST=${EC2_HOST}"
  HOST_IP="$EC2_HOST"
else
  echo "ERROR: cannot start EC2 via API and EC2_HOST is unset." >&2
  echo "Start i-0123fbf60559bd082 in AWS console (eu-west-2), then set EC2_HOST to its public IP." >&2
  exit 2
fi

echo "EC2 public IP: ${HOST_IP}"
REMOTE="${EC2_USER}@${HOST_IP}"

echo "=== Waiting for SSH ==="
for i in $(seq 1 36); do
  if ssh -i "$KEY" -o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new \
    "$REMOTE" 'echo ssh_ok' 2>/dev/null; then
    break
  fi
  sleep 5
  if [[ "$i" -eq 36 ]]; then
    echo "ERROR: SSH never came up on ${HOST_IP}" >&2
    if [[ "$STARTED_VIA_API" -ne 1 ]]; then
      echo "If the instance was stopped, start it in the AWS console first." >&2
    fi
    exit 4
  fi
done

echo "=== Ensuring Docker on VM ==="
ssh -i "$KEY" -o StrictHostKeyChecking=accept-new "$REMOTE" bash -s <<'REMOTE'
set -euo pipefail
if ! command -v docker >/dev/null 2>&1; then
  curl -fsSL https://get.docker.com | sudo sh
  sudo usermod -aG docker "$USER" || true
fi
# Prefer newgrp/docker group, else sudo
if docker info >/dev/null 2>&1; then
  DOCKER="docker"
elif sudo docker info >/dev/null 2>&1; then
  DOCKER="sudo docker"
else
  echo "ERROR: docker installed but not usable" >&2
  exit 1
fi
echo "DOCKER_CMD=$DOCKER"
$DOCKER --version
uname -sm
REMOTE

echo "=== Syncing repo + baking on VM ==="
# Expand ~ on remote for mkdir
ssh -i "$KEY" "$REMOTE" "mkdir -p ${EC2_REPO_PATH}"

# Pass Hub creds via env on the remote command (avoid embedding in heredoc files)
DH_USER="${DOCKERHUB_USERNAME:-}"
DH_TOKEN="${DOCKERHUB_TOKEN:-}"

ssh -i "$KEY" "$REMOTE" \
  "DOCKERHUB_IMAGE=$(printf %q "$DOCKERHUB_IMAGE")" \
  "DOCKERHUB_USERNAME=$(printf %q "$DH_USER")" \
  "DOCKERHUB_TOKEN=$(printf %q "$DH_TOKEN")" \
  "REPO=$(printf %q "$EC2_REPO_PATH")" \
  bash -s <<'REMOTE'
set -euo pipefail
if [[ -d "${REPO}/.git" ]]; then
  cd "${REPO}" && git fetch --depth 1 origin && git reset --hard origin/main || git pull --ff-only || true
else
  git clone https://github.com/arcblanc/Thermoswitches-MLBIO.git "${REPO}"
fi
cd "${REPO}"
chmod +x scripts/build_push_eva_docker.sh
export DOCKERHUB_IMAGE
export EVA_SRC_DIR="${HOME}/EVA"

if docker info >/dev/null 2>&1; then
  :
elif sudo docker info >/dev/null 2>&1; then
  # Wrapper so build script's "docker" calls work when only root can talk to the daemon
  mkdir -p "$HOME/bin"
  cat > "$HOME/bin/docker" <<'WRAP'
#!/bin/sh
exec sudo /usr/bin/docker "$@"
WRAP
  chmod +x "$HOME/bin/docker"
  export PATH="$HOME/bin:$PATH"
fi

if [[ -n "${DOCKERHUB_USERNAME}" && -n "${DOCKERHUB_TOKEN}" ]]; then
  echo "${DOCKERHUB_TOKEN}" | docker login -u "${DOCKERHUB_USERNAME}" --password-stdin
  bash scripts/build_push_eva_docker.sh --push
else
  echo "WARNING: DOCKERHUB_USERNAME/TOKEN not set — build only (no push)."
  echo "Re-run with those env vars, or docker login on the VM, then:"
  echo "  bash scripts/build_push_eva_docker.sh --push"
  bash scripts/build_push_eva_docker.sh
fi
REMOTE

echo "=== Done ==="
echo "If push succeeded, set RunPod Container Image to: ${DOCKERHUB_IMAGE}"
echo "Optional: stop the EC2 instance from the AWS console when finished."
