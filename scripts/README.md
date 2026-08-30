# Scripts layout

Runnable entry points live under category folders. Shared library code stays in [`src/`](../src/).

| Folder | Purpose | Examples |
|--------|---------|----------|
| [`rf/`](rf/) | Random Forest diagnostics, panel merges, XGB ablations | `rf_length_bias_diagnostics.py`, `merge_rus_fused_panel.py` |
| [`eva/`](eva/) | EVA install, Docker bake, cloud batch, smoke tests | `eva_cloud_batch.py`, `macleod_eva_install_smoke.sh` |
| [`triage/`](triage/) | Streaming yield triage and yield-ratio reporting | `eva_stream_triage.py`, `eva_yield_ratio.py` |
| [`extraction/`](extraction/) | Reference downloads and novelty search pipelines | `download_rfam.sh`, `run_novelty_search.sh` |
| [`cloud/`](cloud/) | EC2 / RunPod / S3 thermo batch, Mac local/resume, SSH helpers | `thermo_s3_batch.py`, `runpod_ssh.sh` |
| [`generation/`](generation/) | GenerRNA + BiRNA cloud/smoke path (non-EVA baseline) | `llm_cloud_batch.py`, `llm_smoke_test.sh` |
| [`dev/`](dev/) | Detached launchers, pytest shims, notebook export helpers | `launch_detached.py`, `test_noncircular_features.py` |

Formal tests live in [`tests/`](../tests/) at the repo root. Run `make test` or `uv run pytest tests/ -v`.

## Conventions

- Run from the **repo root** unless a script says otherwise.
- Python scripts prepend `src/` automatically when the project is installed (`uv sync`); legacy `PYTHONPATH=src` still works.
- Shell wrappers set `ROOT` two levels up from `scripts/<category>/`.
- Prefer `PYTHONPATH=src python scripts/<category>/…` in docs and cluster playbooks.

## Quick examples

```bash
# RF diagnostics
PYTHONPATH=src python scripts/rf/rf_length_bias_diagnostics.py --noncircular

# EVA stream triage (Macleod)
PYTHONPATH=src python scripts/triage/eva_stream_triage.py --source ssh

# RefSeq + Rfam extraction
bash scripts/extraction/download_refseq_genomes.sh
bash scripts/extraction/run_novelty_search.sh

# Cloud thermo batch (EC2)
bash scripts/cloud/thermo_ec2_run.sh train --run
```

`_repo_paths.py` exposes `REPO_ROOT` / `SRC_ROOT` for new scripts.
