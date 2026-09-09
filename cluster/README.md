# Cluster runbooks

HPC and historical cloud notes for EVA generation and thermo batches. The root README stays short. Details live here.

## Start here (thesis path)

| Doc | When to read |
|-----|----------------|
| [`EVA_OVERVIEW.md`](EVA_OVERVIEW.md) | Architecture of the frozen EVA 1.4B job, TaxID `mRNA` conditioning, artefact layout |
| [`MACLEOD_EVA.md`](MACLEOD_EVA.md) | Day-to-day Macleod `gpu02` install, tmux, generation, pull-back |
| [`macleod_log.md`](macleod_log.md) | Chronological operator log (what was tried on the cluster) |

Local triage after FASTA lands on the analysis machine:

```sh
PYTHONPATH=src python scripts/triage/eva_stream_triage.py --source ssh
```

Scripts: [`scripts/eva/`](../scripts/eva/), [`scripts/triage/`](../scripts/triage/). See also [`docs/cookbook.md`](../docs/cookbook.md) §4–5.

## Historical / peripheral cloud notes

Kept for appendix reproducibility. Not required for the thesis EVA path.

| Doc / script | Role |
|--------------|------|
| [`CLOUD_PIPELINE.md`](CLOUD_PIPELINE.md) | Older EC2 / S3 thermo batch overview |
| [`README-aws.md`](README-aws.md) | AWS-oriented notes |
| [`EVA_RUNPOD.md`](EVA_RUNPOD.md), [`EVA_SETUP_OVERVIEW.md`](EVA_SETUP_OVERVIEW.md), [`EVA_SETUP_LOG.md`](EVA_SETUP_LOG.md) | RunPod EVA experiments |
| `runpod_*.sh`, `ec2_bootstrap_thermo.sh` | Legacy bootstrap helpers |

## Conventions

- Prefer `tmux` over `nohup` for long GPU jobs (see [`FAQ.md`](../FAQ.md) §9).
- Pull FASTA / logs to the laptop. Do not commit multi-GB weights.
- Yield gates and melting audits run **off** the GPU node after generation.
