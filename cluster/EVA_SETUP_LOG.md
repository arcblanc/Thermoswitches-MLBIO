# EVA setup log — RunPod bake then Macleod GPU

Narrative of what was asked, what broke, and what was fixed. **Not** a captured `docker build` / `pip` stdout dump (those were never teed to `logs/`). Sources: operator runbooks, [`macleod_log.md`](macleod_log.md), and chats [EVA RunPod smoke](fe73f06b-a0c2-49c4-87ae-bb6069bd787b) / [RunPod AWS pipeline](d7914121-414c-479c-8b6d-272b481ed048).

High-level recap after this file: [`EVA_SETUP_OVERVIEW.md`](EVA_SETUP_OVERVIEW.md). Architecture of the **final** stack: [`EVA_OVERVIEW.md`](EVA_OVERVIEW.md).

---

## 0. What you asked for

1. **Replace GenerRNA** with GENTEL-Lab EVA for de novo RNA, smoke-first, then a 10k panel.
2. **Bake a Docker image** (“layer cake”: CUDA PyTorch, flash-attn, MegaBlocks) and **push to Docker Hub** so RunPod can pull `arcblanc/eva-model:v1`.
3. **Run on GPU** — first planned as RunPod Thermopod; generation actually completed on **University Macleod `gpu02`**.
4. Conditioning locked as **`--rna_type mRNA` + `--taxid`** (Option B: E. coli 562 / Salmonella 28901 / Listeria 1639). Never `sRNA`.

---

## 1. RunPod path (Jul 2026) — scaffolding, then bake

### 1.1 Pipeline code (no live generate yet)

Asked: implement EVA smoke + 512-chunk batch on RunPod, hard-fail quality gates, BiRNA, S3, auto-stop pod.

Shipped: `eva_quality.py`, `eva_generate.py`, `eva_cloud_batch.py`, `cluster/runpod_eva_thermopod.sh`, [`EVA_RUNPOD.md`](EVA_RUNPOD.md).

**Design choice that later bit us:** hard batch gate — *any* repetitive/length/format fail aborted the **whole chunk** and would terminate the pod. Fine as a cloud spend fail-safe; fatal on a 6 h Macleod 512.

### 1.2 “Bake the layer cake” (user request, 2026-07-14)

Asked: `docker build -t …/eva-model:v1` then `docker push`, then RunPod pulls the image.

**First error (policy, not compile):** do **not** bake CUDA/`flash-attn` on the M3. Apple Silicon cannot compile the official linux/amd64 EVA Dockerfile. Script `build_push_eva_docker.sh` **hard-fails on Darwin**.

Locked split:

| Machine | Role |
|---------|------|
| M3 Mac | Vienna/NUPACK + SSH only |
| Linux x86_64 VM (`aws-thermo-ec2`) | clone EVA → docker build → push Hub |
| Docker Hub | `arcblanc/eva-model:v1` |
| RunPod | pull image, generate |

### 1.3 Bake VM errors

| Symptom | Cause | Fix |
|---------|--------|-----|
| SSH timeout to `EC2_HOST` | Instance stopped | Start `i-0123fbf60559bd082` in console (`eu-west-2`); IP may change |
| IAM `Arcblanc` cannot `ec2:DescribeInstances` / `StartInstances` | Least-privilege user | Manual console start; `scripts/eva/start_eva_bake_vm.sh` cannot start the VM via API |
| **Disk full during `docker build`** | 25 GB root + CUDA devel + compiling flash-attn | Resize EBS **≥80 GB**, `growpart` + `resize2fs` |
| flash-attn compile too heavy | Source build on small disk | Switch helper to **prebuilt flash-attn wheel** (`build_push_eva_docker.sh`) |
| No Hub token in env | Push skipped | Build-only until `DOCKERHUB_USERNAME` / `DOCKERHUB_TOKEN` set |

Runbook still in [`EVA_RUNPOD.md`](EVA_RUNPOD.md). **No `logs/*.log` of this bake** — output lived on the VM / Cursor terminals.

### 1.4 Why generation moved off RunPod

Scaffolding assumed RunPod + S3 + pod terminate. Live EVA generate used **Macleod** instead: university GPU, `STORAGE_TARGET=local`, `RUNPOD_SKIP_TERMINATE=1`. S3 keys on cluster later failed (`InvalidAccessKeyId`). GenerRNA’s earlier RunPod image (`runpod/pytorch:…`) is a **different** stack than EVA’s MoE/`flash-attn` bake.

---

## 2. Macleod GPU (Aug 2026) — install then generate

Asked: get EVA running on `gpu02` MIG `3g.20gb` (~19.5 GiB; EVA wants ≥16 GB). Session notebook: [`macleod_log.md`](macleod_log.md). Operator copy-paste: [`MACLEOD_EVA.md`](MACLEOD_EVA.md).

### 2.1 Access / paste mistakes

| Symptom | Cause | Fix |
|---------|--------|-----|
| `ssh … 127.0.0.1:1024` from **macleod1** → Connection refused | That port is the **Mac F5 forward** | Only SSH that way **from the Mac** |
| `bash: syntax error` / `command not found` on gpu02 | Pasted chat/transcripts into the GPU shell | Paste **commands only**; results go in the log |
| `tmux: command not found` | tmux is a **module** | `module load tmux` on macleod1 |
| `tmux attach -t eva` on **gpu02** → `no sessions` | tmux lives on **macleod1** | Never attach from inside `srun` |

### 2.2 Python / CUDA bake on the node (not Docker)

Docker image was the RunPod plan. On Macleod we **compiled in a venv** on the MIG.

| Symptom | Cause | Fix |
|---------|--------|-----|
| `GLIBC_2.28 not found` | `gpu02` is CentOS 7, **glibc 2.17**; modern conda Python needs 2.28 | Abandoned conda (`torch_mig` / `torch_el7`). Module `python/3.11.9` + venv `~/sharedscratch/venvs/torch_mod` |
| `ModuleNotFoundError: megablocks` | `pip install -e ~/EVA` is not the full MoE stack | Install `grouped_gemm`, `megablocks`, `stanford-stk`, then flash-attn |
| CUDA extension compile fail | System **GCC 4.8** too old; **GCC 14.2** too new for CUDA 12.4 nvcc | `module load gcc/12.3.0`; `CC`/`CXX`/`CUDAHOSTCXX` |
| `flash-attn` install interrupted | SSH drop (`Connection to 127.0.0.1 closed by remote host`); half-build in `/tmp` **does not persist** | tmux on **macleod1**; re-run `MAX_JOBS=2 pip install flash-attn==2.6.3 --no-build-isolation` |
| Checkpoint path | HF download root is a **tree of sizes** | Smoke must use `~/eva_checkpoint/EVA_1.4B_CLM` not the parent dir |

**Done ~2026-08-11 15:35:** `flash-attn-2.6.3`, 4-seq `eva-generate` smoke FASTA under `~/eva_smoke_out/smoke_ecoli.fa`. Torch **2.5.1+cu124** (EVA pins `<2.6`).

### 2.3 Repo sync

First `rsync` pulled `.tools/infernal` + `models/genererna` — **Ctrl-C**. Lean exclude (`.tools` / `models` / `bin` / `.venv`) then worked.

### 2.4 Orchestrator errors (the expensive ones)

**S3 probe:** `STORAGE_TARGET=s3` → `InvalidAccessKeyId` on `ListObjectsV2`. Fallback: `STORAGE_TARGET=local` + `--skip-s3-probe`.

**32-seq smoke, T=0.8 — QUALITY GATE FAILED**

```text
gate_stats: ok=False … categories={'Repetitive Text Collapse': 1}
seq[5] unique_3mer_ratio=0.048 < 0.05
```

Retry **T=0.9** → **32/32** passed. Many seqs hit **600 nt** max length.

**512 E. coli, one 512-seq CLI (~6 h) — QUALITY GATE FAILED**

```text
accepted=512 errors=3  Repetitive Text Collapse
seq[152] 0.045 < 0.05
seq[243] 0.028 < 0.05
seq[433] 0.048 < 0.05
```

**509/512 would have passed.** Hard abort wrote **nothing** to `generated.fasta` (temp chunk gone).

**Fix:** raw-persist → **soft-drop** (keep passers) → checkpoint → top-up. Chunk **128** so a kill loses ~1.5 h not 6 h.

**Re-run:** 512/512 accepted, 6 soft-dropped. Mac yield **31/512 = 6.05%** (Path A).

### 2.5 2k top-up + Mac stream triage

`--resume` to 2000. Stream watcher on Mac.

| Symptom | Cause | Fix |
|---------|--------|-----|
| rsync `change_dir "/Users/amierzuhri/Thermoswitches-MLBIO/..."` on **cluster** | Mac shell expanded `~/` in `--remote-fasta` | Absolute `/home/t41am25/Thermoswitches-MLBIO/...` |
| `BatchMode=yes` / password | Cluster is password auth | ControlMaster; no BatchMode |
| `Connection refused` `:1024` | F5 tunnel down | Watcher polls; reconnect F5. Old code aborted after 3 failures |
| `pending=238` forever at 2000 | Stride 250; leftover < stride | `--once --flush-remainder` (slice_006) |
| Mac `tmux` not found | Homebrew tmux never installed | Leave Cursor terminal open, or `brew install tmux` |

**Finished:** FASTA **2000/2000**. Stream 74/1488 + pilot 31/512 → **105/2000 = 5.25%**. `squeue` empty; tmux `eva` / `eva2k` killed.

---

## 3. Error cheat sheet (same as operators learned)

| Error | Where | Fix |
|-------|--------|-----|
| Darwin bake refused | Mac | SSH to linux/amd64 VM |
| Disk full / flash-attn OOM disk | EC2 bake | ≥80 GB root + prebuilt wheel |
| IAM cannot start EC2 | Mac AWS user | Console start instance |
| `GLIBC_2.28 not found` | gpu02 conda | Module Python + venv |
| `megablocks` missing | first eva-generate | CUDA MoE deps + GCC 12.3 |
| flash-attn half-built | SSH drop | tmux on macleod1; re-pip |
| `InvalidAccessKeyId` | cluster S3 | local storage + skip probe |
| QUALITY GATE FAILED (3-mer) | hard batch gate | soft-drop + T=0.9 + chunk 128 |
| F5 / rsync `~/` | Mac triage | absolute remote path; keep tunnel |
| tmux on gpu02 | overlap `srun` | attach from macleod1 only |

---

## 4. What we never captured

- Docker bake stdout on `aws-thermo-ec2`
- `pip install flash-attn` full compile log (only the success line in `macleod_log.md`)
- tmux pane history after `tmux kill-session`

Those lives were terminal sessions, not `logs/`. This file is the reconstructed log.
