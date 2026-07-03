#!/usr/bin/env bash
# One-time bootstrap for aws-thermo-ec2 (c7i-flex.large, 2 vCPU).
# Run over SSH on the already-running instance.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

sudo apt-get update
sudo apt-get install -y python3-venv python3-pip

python3 -m venv .venv
source .venv/bin/activate
pip install -q -r requirements.txt
pip install -q -r requirements-aws.txt
pip install -q viennarna joblib

WHEEL="$(python -c "
import sys
from pathlib import Path
sys.path.insert(0, 'src')
from thermo_sim.thermo_common import local_nupack_wheel_path
p = local_nupack_wheel_path()
print(p or '')
")"

if [[ -n "$WHEEL" && -f "$WHEEL" ]]; then
  pip install "$WHEEL"
  echo "Installed NUPACK from $WHEEL"
else
  echo "WARNING: NUPACK wheel not found under nupack-4.1.0.1/package/"
  echo "SCP the licensed Linux wheel to the instance, then re-run pip install."
fi

python src/thermo_sim/vienna_rna.py --dry-run
python src/thermo_sim/nupack_engine.py --dry-run
echo "Bootstrap complete."
