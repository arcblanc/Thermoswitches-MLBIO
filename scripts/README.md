# Scripts

Thin orchestration CLIs. Reusable logic stays in [`src/`](../src/). Formal tests: [`tests/`](../tests/) (`make test`).

## Layout

| Folder | Purpose | Examples |
|--------|---------|----------|
| [`rf/`](rf/) | RF diagnostics, panel merges, XGB ablations | `rf_length_bias_diagnostics.py`, `merge_rus_fused_panel.py` |
| [`eva/`](eva/) | EVA install, Docker bake, cloud batch, smoke tests | `eva_cloud_batch.py`, `macleod_eva_install_smoke.sh` |
| [`triage/`](triage/) | Streaming yield triage and yield-ratio reporting | `eva_stream_triage.py`, `eva_yield_ratio.py` |
| [`extraction/`](extraction/) | Reference downloads and novelty search | `download_rfam.sh`, `run_novelty_search.sh` |
| [`cloud/`](cloud/) | EC2 / RunPod / S3 thermo batch, SSH helpers | `thermo_s3_batch.py`, `runpod_ssh.sh` |
| [`generation/`](generation/) | GenerRNA + BiRNA cloud/smoke (appendix path) | `llm_cloud_batch.py`, `llm_smoke_test.sh` |
| [`dev/`](dev/) | Detached launchers, pytest shims, export helpers | `launch_detached.py`, `test_noncircular_features.py` |

## Conventions

- Run from the **repo root** unless a script says otherwise.
- After `uv sync`, prefer `uv run python scripts/<category>/…`. Legacy `PYTHONPATH=src python …` still works and is what cluster docs use.
- Shell wrappers set `ROOT` two levels up from `scripts/<category>/`.
- `_repo_paths.py` exposes `REPO_ROOT` / `SRC_ROOT` for new scripts.

## Quick examples

```sh
# RF diagnostics (noncircular honesty path)
PYTHONPATH=src python scripts/rf/rf_length_bias_diagnostics.py --noncircular

# EVA stream triage after Macleod generation
PYTHONPATH=src python scripts/triage/eva_stream_triage.py --source ssh

# RefSeq download + Rfam novelty search
bash scripts/extraction/download_refseq_genomes.sh
bash scripts/extraction/run_novelty_search.sh

# Cloud thermo batch (historical EC2 path)
bash scripts/cloud/thermo_ec2_run.sh train --run
```

Marimo thesis apps: [`notebooks/README.md`](../notebooks/README.md). Full rebuild recipes: [`docs/cookbook.md`](../docs/cookbook.md).
