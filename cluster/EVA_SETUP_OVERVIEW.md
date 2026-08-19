# EVA setup overview

What the EVA path is **now**, after the errors in [`EVA_SETUP_LOG.md`](EVA_SETUP_LOG.md). For trees and tmux reasons see [`EVA_OVERVIEW.md`](EVA_OVERVIEW.md). For copy-paste commands see [`MACLEOD_EVA.md`](MACLEOD_EVA.md).

---

## Two attempts, one finish

```
Plan A — RunPod
  M3 SSH  →  Linux VM docker bake  →  Docker Hub  →  RunPod pull  →  S3
  Status:  scaffolding + bake scripts; live 2k generate did not run here

Plan B — Macleod (what actually produced sequences)
  F5  →  macleod1 tmux  →  srun gpu02 MIG  →  venv + EVA 1.4B CLM
       →  generated.fasta  →  Mac Vienna + Rfam novelty
  Status:  2000/2000 accepted; yield 105/2000 = 5.25%
```

EVA weights are **pretrained** (GENTEL-Lab). We baked a **runtime** (Docker for RunPod; venv+CUDA deps on Macleod), then wrapped `eva-generate` with chunking and filters.

---

## Why Plan A stalled / moved

| Intent | What happened |
|--------|----------------|
| Bake on the Mac | Refused — CUDA/flash-attn is linux/amd64 only |
| Bake on `aws-thermo-ec2` | IAM could not start the instance; 25 GB disk filled compiling flash-attn |
| RunPod generate | Code ready (`EVA_RUNPOD.md`); S3/keys and university GPU took over |
| Hard quality gate | Designed to kill a paid pod; later wiped a 6 h 512 on Macleod |

---

## Why Plan B worked (after the same class of errors)

| Layer | Final choice |
|-------|----------------|
| Host | `gpu02` MIG `3g.20gb` (glibc 2.17 → **module Python**, not conda) |
| Env | `torch_mod` + CUDA 12.4 + **GCC 12.3** + megablocks + flash-attn 2.6.3 |
| Job survival | **tmux on macleod1**, not gpu02 |
| Storage | **local** FASTA (`InvalidAccessKeyId` on cluster S3) |
| Filters | **soft-drop** + persist raw + chunk **128** + T=**0.9** |
| Triage | Mac CPU; absolute SSH path; `--flush-remainder` at the end |

---

## Numbers that matter

| Run | Result |
|-----|--------|
| 4-seq CLI smoke | OK |
| 32-seq orchestrator T=0.8 | Hard-gate fail (1 seq 3-mer) |
| 32-seq T=0.9 | 32/32 |
| 512 hard-gate | 3 seqs fail → **0 written** after ~6 h |
| 512 soft-drop | 512/512 (6 dropped); yield **6.05%** |
| 2000 resume | 2000/2000; combined yield **5.25%** |

---

## Read next

1. This overview (you are here)
2. [`EVA_SETUP_LOG.md`](EVA_SETUP_LOG.md) — every error and the ask that triggered it
3. [`EVA_OVERVIEW.md`](EVA_OVERVIEW.md) — architecture tree
4. [`MACLEOD_EVA.md`](MACLEOD_EVA.md) / [`EVA_RUNPOD.md`](EVA_RUNPOD.md) — commands
