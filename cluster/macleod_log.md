# Macleod GPU access log

Session notes from connecting to the University Macleod cluster (`t41am25@macleod1` → `gpu02`), proving CUDA on a MIG slice, and starting an EVA install/smoke.

**Dates:** 2026-08-11 (access + EVA install)  
**User:** `t41am25`  
**Login node:** `macleod1.cluster.local`  
**SSH (from laptop):** `ssh -l t41am25 -p 1024 127.0.0.1` (F5 VPN local forward; password in `.env` as `MACLEOD_PSSWRD`)  
**GPU target:** `gpu02` — Slurm GRES `gpu:3g.20gb` (A100 MIG ~20 GB slices)  
**Working Python stack (current):** module `python/3.11.9` + venv `~/sharedscratch/venvs/torch_mod` (not conda)  
**Status:** 2k generation **2000/2000** done; Mac stream triage **finished** (watermark 2000). Combined yield **105/2000 = 5.25%**. Overview: [`EVA_OVERVIEW.md`](EVA_OVERVIEW.md).

Jump to: [Session 2 summary](#session-2-summary-2026-08-11-night--eva-install-in-progress) · [Paste rules](#how-to-paste-into-this-log-and-into-the-live-terminal) · [Status board](#status-board)

---

## Cluster layout (context)

| Node | GRES | Meaning |
|------|------|---------|
| `gpu01` | `gpu:1g.5gb:21` | Many small ~5 GB MIG slices |
| `gpu02` | `gpu:3g.20gb:6` | Six ~20 GB MIG slices (this session) |

Partition: `gpu`. Login node has **no** GPU.

---

## 1. Login node — who/where and GPU presence

```bash
hostname; whoami; nvidia-smi -L 2>/dev/null || echo "no GPU on login node"
```

**Does:** Prints machine name and account; lists GPUs if `nvidia-smi` works.  
**Checks:** You are on the login node as `t41am25`, not on a GPU node. Expected: `macleod1.cluster.local`, `no GPU on login node`.

---

## 2. Cluster inventory — nodes, GRES, state

```bash
sinfo -N -o "%N %G %t %c %m" 2>/dev/null | head -50
```

**Does:** One line per node: name, GRES (GPU type), state, CPUs, memory.  
**Checks:** Which compute/GPU nodes exist and whether they are `idle` / `alloc` / `mix`. Confirmed `gpu01` / `gpu02` and that `gpu02` advertises `gpu:3g.20gb:6`.

```bash
sinfo -p gpu 2>/dev/null; sinfo 2>/dev/null | head -40
```

**Does:** Shows the `gpu` partition, then a short full-partition summary.  
**Checks:** Partition `gpu` is `up`, both `gpu[01-02]` idle at the time of the session.

---

## 3. Inspect `gpu02` in detail

```bash
scontrol show node gpu02 2>/dev/null
```

**Does:** Full Slurm node record (CPUs, memory, GRES, TRES, state, boot time).  
**Checks:** Exact GRES string (`gpu:3g.20gb:6`), 48 CPUs, ~250 GB RAM, partition membership `gpu`, idle/free resources.

```bash
sinfo -N -n gpu02 -o "%N %G %t %c %m %f"
```

**Does:** Compact one-liner for `gpu02` only.  
**Checks:** Same facts in a short table (GRES / state / CPUs / memory / features).

---

## 4. Wrong GRES request (failure)

```bash
srun -w gpu02 --gres=gpu:1 --pty bash -i
```

**Does:** Asks Slurm for an interactive shell on `gpu02` with generic `gpu:1`.  
**Checks:** Whether bare `gpu:1` matches the node.  
**Result:** `Requested node configuration is not available` — Slurm needs the **MIG profile name**, not a generic GPU count.

---

## 5. Correct interactive job on `gpu02` (success)

```bash
srun -p gpu -w gpu02 --gres=gpu:3g.20gb:1 --pty bash -i
```

**Does:** Interactive bash on partition `gpu`, pinned to `gpu02`, one `3g.20gb` MIG device.  
**Checks:** You can allocate a 20 GB slice and land on the GPU node.  
**Result:** Job queued then allocated (e.g. job `322695`); prompt becomes `[t41am25@gpu02 ~]$`.

---

## 6. On `gpu02` — hardware visibility

```bash
hostname
```

**Does:** Confirms hostname.  
**Checks:** Shell is on `gpu02.cluster.local`, not the login node.

```bash
nvidia-smi -L
```

**Does:** Lists physical GPUs and visible MIG devices.  
**Checks:** Three A100-PCIE-40GB present; at least one MIG `3g.20gb` device UUID visible to the job.

```bash
nvidia-smi
```

**Does:** Driver/CUDA version, per-GPU power/memory, MIG table, processes.  
**Checks:** Driver **550.127.05**, CUDA **12.4**, MIG enabled, allocated MIG ~**19968 MiB** (~20 GB), no other user processes on the slice.

```bash
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
```

**Does:** Prints which GPU index Slurm exposed to the job.  
**Checks:** Typically `CUDA_VISIBLE_DEVICES=0` (the allocated MIG appears as device 0 inside the job).

---

## 7. System Python / modules (no Torch yet)

```bash
module avail cuda 2>/dev/null | head
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory/1024**3)"
```

**Does:** Tries to list CUDA modules; tries Torch on default `python3`.  
**Checks:** Whether a ready CUDA/Torch stack exists. First `module avail cuda` looked empty without full `module avail`; system Torch import hung / failed — wrong Python.

```bash
which python3
python3 -V
python3 -c "print('ok')"
python3 -c "import torch; print('torch', torch.__version__)"
```

**Does:** Locates system Python and tests import.  
**Checks:** `/usr/bin/python3` is **3.6.8**; `print('ok')` works; **`No module named 'torch'`**.

```bash
module avail 2>&1 | head -80
ls ~/miniconda*/bin/conda ~/anaconda*/bin/conda ~/.conda 2>/dev/null
conda env list 2>/dev/null
```

**Does:** Lists Lmod modules; looks for a personal conda install.  
**Checks:** Available software (notably `cuda/12.4.0`, `cuda/12.6.2`, `miniconda3`, `python/3.11.9`, `singularity/3.8.5`). No personal conda under `~/` yet.

---

## 8. Load modern Python + CUDA modules

```bash
module load miniconda3
module load cuda/12.4.0
module load python/3.11.9

which python
python -V
python -c "print('ok')"
```

**Does:** Loads cluster Miniconda, CUDA 12.4, Python 3.11.9; verifies the active interpreter.  
**Checks:** `python` points at Spack/module **3.11.9**, not system 3.6.

---

## 9. Create env and install PyTorch (CUDA 12.4)

```bash
conda create -y -n torch_mig python=3.11
conda activate torch_mig
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

**Does:** New conda env; installs PyTorch built for **cu124** (matches node CUDA 12.4 / driver 550).  
**Checks:** Env can install large GPU wheels.  
**Result:** `torch-2.6.0+cu124` and NVIDIA CUDA 12.4 pip libs installed successfully.

---

## 10. Prove CUDA in PyTorch (MIG)

Default `CUDA_VISIBLE_DEVICES=0` first-load could hang; using the MIG UUID was reliable:

```bash
export CUDA_VISIBLE_DEVICES=MIG-bd9dac73-75f8-5071-9a48-bcfc002e3d2c
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

**Does:** Restricts Torch to the allocated MIG UUID; reports CUDA availability and device count.  
**Checks:** **`True 1`** — Torch sees exactly one CUDA device.

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.get_device_name(0)); print(round(torch.cuda.get_device_properties(0).total_memory/1024**3, 2), 'GiB')"
```

**Does:** Prints Torch build, device name, VRAM.  
**Checks:** `2.6.0+cu124`, name **`NVIDIA A100-PCIE-40GB MIG 3g.20gb`**, **`19.5 GiB`**.

---

## 11. Compute smoke (alloc + matmul)

Large 2048² matmul appeared stuck (likely first cuBLAS warm-up). Smaller probes:

```bash
python -c "import torch; print('start'); a=torch.zeros(1,device='cuda'); print('alloc_ok', a); torch.cuda.synchronize(); print('sync_ok')"
```

**Does:** Allocates a 1-element tensor on GPU and synchronizes.  
**Checks:** Device memory allocation and CUDA runtime sync work (`alloc_ok` / `sync_ok`).

```bash
python -c "import torch; x=torch.randn(512,512,device='cuda'); y=x@x; torch.cuda.synchronize(); print('matmul_ok', float(y.mean()))"
```

**Does:** Random GEMM on GPU.  
**Checks:** Actual compute kernels run (not just device enumeration).  
**Result:** `matmul_ok` with a numeric mean — **GPU compute verified**.

---

## Outcome (session 1 — GPU access)

| Check | Result |
|-------|--------|
| SSH to Macleod (F5 → `127.0.0.1:1024`) | OK (`macleod1`) |
| Slurm `gpu` partition / `gpu02` | OK |
| Interactive MIG `3g.20gb:1` | OK |
| `nvidia-smi` | 3× A100, MIG ~20 GB |
| System `/usr/bin/python3` | 3.6 — no Torch |
| First CUDA proof | OK via temporary conda `torch_mig` + Torch 2.6.0+cu124, **19.5 GiB**, matmul OK |

---

## How to paste into this log (and into the live terminal)

**This file** (`cluster/macleod_log.md`) is the lab notebook. When you paste terminal history here (or into chat), include:

1. The **commands you typed**
2. The **important output** (success lines, errors)
3. A one-line note of **what it checked**

**Do not paste back into the Macleod shell:**

- Prompt lines like `(torch_mod) [t41am25@gpu02 ~]$`
- Whole `nvidia-smi` ASCII tables
- Mixed “command + old output” blocks  

That causes `bash: syntax error` and wastes the session. Paste **only** the next command block into the terminal; paste **commands + results** into this log / chat.

---

## Session 2 summary (2026-08-11 night) — EVA install in progress

### What you achieved

1. Re-allocated `gpu02` with `srun -p gpu -w gpu02 --gres=gpu:3g.20gb:1` (job ~322696).
2. Discovered **host glibc 2.17** (CentOS 7). Modern conda Python (`torch_mig`, then `torch_el7`) dies with `GLIBC_2.28 not found` — abandoned conda for GPU work.
3. Switched to **module `python/3.11.9` + venv**  
   `$HOME/sharedscratch/venvs/torch_mod`  
   Installed Torch → **`2.5.1+cu124 True 1`** (EVA pins Torch `<2.6`).
4. Cloned **GENTEL-Lab/EVA** to `~/EVA` (`module load git`), `pip install -e .` → `eva-generate --help` works.
5. Downloaded HF weights to `~/eva_checkpoint` (all sizes; use **`EVA_1.4B_CLM`** for CLM smoke).
6. First `eva-generate` failed: missing **`megablocks`** (MoE CUDA stack).
7. Built CUDA extensions after fixing the compiler stack:
   - System GCC 4.8 → too old  
   - GCC 14.2 → too new for CUDA 12.4 nvcc  
   - **GCC 12.3** → works  
   - Installed **`grouped_gemm==0.1.6`**, **`megablocks==0.7.0`**, **`stanford-stk`**, ninja/einops/accelerate  
8. **`flash-attn==2.6.3`** compile was interrupted when SSH closed (`Connection to 127.0.0.1 closed by remote host`). Treat as **not installed** until verified.
9. Smoke generate (**4× mRNA, taxid 562**) not run yet — blocked on `flash-attn`.

### What persists if SSH drops

| Artifact | Path | Persists? |
|----------|------|-----------|
| Venv + pip packages | `~/sharedscratch/venvs/torch_mod` | Yes (once install finishes) |
| EVA source | `~/EVA` | Yes |
| Checkpoints | `~/eva_checkpoint/` | Yes |
| Half-built `flash-attn` | temp build dir | **No** — re-run that pip |
| `srun` GPU job | — | **No** |

Prefer `tmux`/`sbatch` for overnight builds.

### Resume tomorrow (after flash-attn succeeds)

```bash
srun -p gpu -w gpu02 --gres=gpu:3g.20gb:1 --pty bash -i
module load python/3.11.9 cuda/12.4.0 gcc/12.3.0 git
source $HOME/sharedscratch/venvs/torch_mod/bin/activate
export CUDA_VISIBLE_DEVICES=0
export CUDA_MODULE_LOADING=LAZY
export CC=$(which gcc) CXX=$(which g++) CUDAHOSTCXX=$(which g++)
export TORCH_CUDA_ARCH_LIST=8.0

# if flash-attn did not finish last night:
# MAX_JOBS=2 pip install flash-attn==2.6.3 --no-build-isolation

python -c "import torch, flash_attn, megablocks, grouped_gemm; print(torch.__version__, torch.cuda.is_available())"

mkdir -p $HOME/eva_smoke_out
eva-generate \
  --checkpoint $HOME/eva_checkpoint/EVA_1.4B_CLM \
  --format clm --rna_type mRNA --taxid 562 --num_seqs 4 \
  --output $HOME/eva_smoke_out/smoke_ecoli.fa
head -n 20 $HOME/eva_smoke_out/smoke_ecoli.fa
```

---

## Session 2 — commands (detail)

### A. glibc / conda dead end

```bash
ldd --version | head -1          # → glibc 2.17
conda env remove -y -n torch_mig
conda create -y -n torch_el7 -c defaults python=3.10
# python -V → GLIBC_2.28 not found  (conda Python unusable on this node)
```

**Checks:** Host libc age vs modern Anaconda builds.  
**Lesson:** use module Python + venv, not new conda envs, on `gpu02`.

### B. Working Torch stack

```bash
module unload miniconda3 2>/dev/null || true
module load python/3.11.9 cuda/12.4.0
python -m venv $HOME/sharedscratch/venvs/torch_mod
source $HOME/sharedscratch/venvs/torch_mod/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu124
# after eva-rna install, Torch is 2.5.1+cu124 (pinned by EVA)
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
# → 2.5.1+cu124 True 1
```

### C. EVA clone + package

```bash
module load git
git clone https://github.com/GENTEL-Lab/EVA.git $HOME/EVA
cd $HOME/EVA
pip install -e .
eva-generate --help | head
```

**Checks:** CLI entry point on PATH inside the venv.

### D. Checkpoint download

```bash
huggingface-cli download GENTEL-Lab/EVA --local-dir $HOME/eva_checkpoint
# completed: EVA_1.4B_CLM, EVA_1.4B_GLM, EVA_437M, EVA_145M, EVA_21M, …
```

**Checks:** Weights on disk (~multi‑GB). Smoke must use a **subdir**, e.g. `$HOME/eva_checkpoint/EVA_1.4B_CLM`.

### E. First generate failure

```bash
eva-generate --checkpoint $HOME/eva_checkpoint/EVA_1.4B_CLM \
  --format clm --rna_type mRNA --taxid 562 --num_seqs 4 \
  --output $HOME/eva_smoke_out/smoke_ecoli.fa
# → ModuleNotFoundError: megablocks
```

**Checks:** Lightweight `pip install -e .` is not enough for 1.4B MoE; need Docker-stack CUDA deps.

### F. MoE CUDA deps (compiler dance)

```bash
# FAIL: system gcc 4.8 / or gcc 14.2 with CUDA 12.4
module load gcc/12.3.0
export CC=$(which gcc) CXX=$(which g++) CUDAHOSTCXX=$(which g++)
export TORCH_CUDA_ARCH_LIST=8.0

pip install ninja packaging einops accelerate
pip install stanford-stk==0.7.1
pip install --no-build-isolation grouped_gemm==0.1.6   # OK
pip install --no-build-isolation megablocks==0.7.0     # OK
MAX_JOBS=2 pip install flash-attn==2.6.3 --no-build-isolation  # building when paused
```

**Checks:** Extension compile against A100 (`sm_80`) with a nvcc-compatible host GCC.

---

## Reusable cheat sheet (current best practice)

```bash
# Login → GPU
srun -p gpu -w gpu02 --gres=gpu:3g.20gb:1 --pty bash -i

module load python/3.11.9 cuda/12.4.0 gcc/12.3.0 git
source $HOME/sharedscratch/venvs/torch_mod/bin/activate
export CUDA_VISIBLE_DEVICES=0 CUDA_MODULE_LOADING=LAZY
export CC=$(which gcc) CXX=$(which g++) CUDAHOSTCXX=$(which g++)

python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# Release GPU when done
exit
```

gpu01 (5 GB slices):

```bash
srun -p gpu -w gpu01 --gres=gpu:1g.5gb:1 --pty bash -i
```

More EVA detail: [`MACLEOD_EVA.md`](MACLEOD_EVA.md).

---

## Status board

| Step | Status |
|------|--------|
| F5 SSH / login node | Done |
| MIG GPU alloc + nvidia-smi | Done |
| CUDA Torch on MIG (~19.5 GiB) | Done (`torch_mod` venv) |
| EVA source + `eva-generate` CLI | Done |
| HF checkpoints | Done |
| `grouped_gemm` / `megablocks` | Done (GCC 12.3) |
| `flash-attn` | **Done** (`Successfully installed flash-attn-2.6.3`, 2026-08-11 ~15:35) |
| 4-seq `eva-generate` smoke | **Done** — FASTA written under `~/eva_smoke_out/smoke_ecoli.fa` (headers like `>mRNA_taxid562_seq2_len310`) |
| Repo orchestrator + gate_stats | **Done on cluster** — `gate_stats` printed on real chunk |
| FASTA → dynamic Vienna CLI | **Code ready** — `src/thermo_sim/extract_fasta_dynamic_features.py` |
| Yield ratio + novelty (blastn/nhmmer) | **Done** — pilot + stream slices 001–006 |
| Repo sync Mac → Macleod | **Done** — lean `rsync` to `~/Thermoswitches-MLBIO` (exclude `.tools`/`models`/`bin`) |
| Step 1 gated smoke 32 seq | **Done** — 32/32 FASTA + manifest; multi-TaxID; `T=0.9`; `STORAGE_TARGET=local` |
| Step 2 pilot 512 → local (S3 later) | **Done** (soft-drop re-run) — 512/512; yield **31/512 = 6.05%** |
| Step 2b 2k top-up | **Done** — 2000/2000 accepted |
| Step 3 local Mac triage | **Done** — stream slices 001–006; watermark 2000; combined **105/2000 = 5.25%** |

### Note on `ssh …127.0.0.1` from macleod1

That address/port is the **laptop F5 forward**. From `macleod1` it always says `Connection refused`. Only use `ssh -p 1024 t41am25@127.0.0.1` on your **Mac**.

---

## Session: triage ladder code (2026-08-11)

Implemented locally (sync/pull repo on Macleod before GPU runs):

1. `eva_generate.py` — `evaluate_chunk` + `gate_stats` in stdout/`run_state.json`
2. `extract_fasta_dynamic_features.py` — header-stable Z-score seeds
3. `eva_yield_ratio.py` — no BLAST/nhmmer hit ⇒ `E_Rfam=inf`
4. `run_eva_pilot_novelty.sh` — all FASTA IDs vs Rfam 14.9

Copy-paste Step 1–3: see [`MACLEOD_EVA.md`](MACLEOD_EVA.md) §4.

---

## Session: Step 1 gated smoke on `gpu02` (2026-08-11 ~16:05–16:43)

### Sync

- First full `rsync` dragged `.tools/infernal` + `models/genererna` — **Ctrl-C**.
- Lean `rsync` (exclude `.tools`/`bin`/`models`/`.vscode`) completed; `src/` + `scripts/` present under `~/Thermoswitches-MLBIO`.

### Env (working)

```bash
cd ~/Thermoswitches-MLBIO
# torch_mod venv + modules already loaded in tmux session `eva`
export STORAGE_TARGET=local
export RUNPOD_SKIP_TERMINATE=1
export EVA_CHECKPOINT_DIR=$HOME/eva_checkpoint/EVA_1.4B_CLM
export EVA_NUM_SAMPLES=32
export EVA_CHUNK_SIZE=32
export PYTHONPATH=src:${PYTHONPATH:-}
pip install -q python-dotenv boto3   # done once
```

Dry-run OK: multi-TaxID panel 11+11+10, checkpoint `EVA_1.4B_CLM`, prefix intended `llm-batch/eva/v1/smoke`.

### Attempt 1 — S3

- `STORAGE_TARGET=s3` → `botocore.exceptions.ClientError: InvalidAccessKeyId` on `ListObjectsV2`.
- **Action:** fall back to `STORAGE_TARGET=local` + `--skip-s3-probe` until Mac `aws sts get-caller-identity` works / keys rotated. Do **not** log secret values here.

### Attempt 2 — local generate (`T=0.8`)

```text
=== Host ecoli taxid=562 remaining=11/11 ===
  $ …/eva-generate … --num_seqs 11 --temperature 0.8 …
  gate_stats: ok=False accepted=11 empty_skipped=0 errors=1
    categories={'Repetitive Text Collapse': 1}
QUALITY GATE FAILED — aborting EVA run:
  Repetitive Text Collapse: seq[5] unique_3mer_ratio=0.048 < 0.05
```

- Gates + `gate_stats` path **confirmed working**.
- Failure is borderline low 3-mer diversity (0.048 vs 0.05), not a CUDA/install issue.
- Cleared partial artifacts: `generated.fasta`, `generation_manifest.jsonl`, `run_state.json`.

### Attempt 3 — local retry (`T=0.9`) — **PASSED** (~16:43–16:59)

```bash
python src/de_novo_hallucinations/eva_generate.py --smoke --resume \
  --chunk-size 32 \
  --output-fasta data/processed/de_novo/generated.fasta \
  --skip-s3-probe \
  --temperature 0.9
```

Result:

- E. coli / Salmonella / Listeria chunks all generated.
- Listeria: `gate_stats: ok=True accepted=10 empty_skipped=0 errors=0`
- `Checkpoint: 32/32` → `Wrote 32 new EVA sequences to …/data/processed/de_novo/generated.fasta`
- Verify:
  - `grep -c '^>' data/processed/de_novo/generated.fasta` → **32**
  - `wc -l data/processed/de_novo/generation_manifest.jsonl` → **32**
- Many sequences length **600 nt** (hitting `--max_length` / `EVA_MAX_LEN`). Watch in Step 2.
- `RUNPOD_SKIP_TERMINATE` left pod/job alone (correct on Macleod).

**Step 1 criteria met** (gated smoke, multi-TaxID, local artifacts). S3 upload still deferred until IAM keys work.

---

## Step 2 — Scale pilot 512 (E. coli only, local) — copy-paste

**On `gpu02`**, clear smoke outputs so IDs/manifest don’t mix, then:

```bash
cd ~/Thermoswitches-MLBIO
# env still: torch_mod, CUDA_VISIBLE_DEVICES=0, PYTHONPATH=src, RUNPOD_SKIP_TERMINATE=1
export STORAGE_TARGET=local
export EVA_CHECKPOINT_DIR=$HOME/eva_checkpoint/EVA_1.4B_CLM
export EVA_SMOKE_SINGLE_HOST=1
export EVA_NUM_SAMPLES=512
export EVA_CHUNK_SIZE=512
export EVA_BATCH_SIZE=1

# archive smoke (optional) then clear working artifacts
mkdir -p data/processed/eva_smoke32
mv -f data/processed/de_novo/generated.fasta data/processed/eva_smoke32/ 2>/dev/null || true
mv -f data/processed/de_novo/generation_manifest.jsonl data/processed/eva_smoke32/ 2>/dev/null || true
rm -f data/processed/run_state.json

# optional VRAM watch in another pane: nvidia-smi -l 5
python src/de_novo_hallucinations/eva_generate.py --smoke --resume \
  --chunk-size 512 \
  --output-fasta data/processed/de_novo/generated.fasta \
  --skip-s3-probe \
  --temperature 0.9
```

Expect ~1 chunk, `gate_stats`, then `Wrote 512…`. Wall-clock will be much longer than Step 1. On pass: `grep -c '^>' …/generated.fasta` → **512**.

### Step 2 progress check (2026-08-11 ~18:35)

Second SSH + `srun --overlap` on `gpu02` (monitor only; do **not** `tmux attach` on gpu02 — session is on **macleod1**):

```text
etime ~01:31:40
28805  python …/eva_generate.py --smoke --resume --chunk-size 512 …
28806  …/eva-generate … --num_seqs 512 --temperature 0.9 … /tmp/eva_chunk_mdt9tt20/chunk.fa
```

Follow-up ~19:35 (`etime ~02:31:25`, same PIDs 28805/28806):

- Still alive; same temp path.
- `ls /tmp/eva_chunk_mdt9tt20/chunk.fa` empty / missing from monitor shell — normal if EVA writes the FASTA at end of the CLI call, or `/tmp` isolation across jobs.
- `generated.fasta` not present yet — orchestrator only writes **after** the 512-seq `eva-generate` returns and gates pass.
- `tmux attach -t eva` on **gpu02** → `no sessions` (attach from **macleod1** instead).

- ETA still roughly **several more hours** at `batch_size=1`.

Follow-up ~20:50 (`etime ~03:43:11`, same PIDs): still in the 512-seq CLI; no `generated.fasta` yet. Pace matches ~4 h smoke-based ETA (near the end of that window). **Do not paste chat/transcripts into the gpu02 shell** (causes `command not found` noise; does not kill the job).

### Step 2 result — QUALITY GATE FAILED (~23:06)

`eva-generate` **did finish** 512 sequences (~6 h wall, slower than the pure 30 s/seq smoke extrapolation). Then:

```text
gate_stats: ok=False accepted=512 empty_skipped=0 errors=3
  categories={'Repetitive Text Collapse': 3}
QUALITY GATE FAILED — aborting EVA run:
  seq[152] unique_3mer_ratio=0.045 < 0.05
  seq[243] unique_3mer_ratio=0.028 < 0.05
  seq[433] unique_3mer_ratio=0.048 < 0.05
```

- **509/512** would have passed; hard abort writes **nothing** to `generated.fasta` (temp chunk already gone).
- Same failure mode as Step 1 smoke (borderline 3-mer diversity), now at scale.

**Fix applied (local Mac → rsync):** orchestrator is now **raw-persist + soft-drop + chunk checkpoint**:

1. Every CLI chunk is appended to `data/processed/de_novo/raw_generated.fasta` (+ per-chunk under `raw_chunks/`) **before** filtering.
2. Per-sequence soft filter keeps passers; drop reasons go to stdout / `run_state.json` (`gate_stats.mode=soft_drop`).
3. Only passers append to `generated.fasta` / `generation_manifest.jsonl`; loop tops up until host quota is met (`EVA_SOFT_TOPUP_BUFFER=1.05`).
4. Prefer `EVA_CHUNK_SIZE=128` so a kill loses ≤~1.5 h instead of ~6 h.

### Step 2 re-run (soft-drop, 128-seq chunks)

On Mac: rsync lean repo to cluster, then on `gpu02`:

```bash
cd ~/Thermoswitches-MLBIO
module load python/3.11.9 cuda/12.4.0 gcc/12.3.0
source ~/sharedscratch/venvs/torch_mod/bin/activate
export CUDA_VISIBLE_DEVICES=0 CUDA_MODULE_LOADING=LAZY
export PYTHONPATH=src:${PYTHONPATH:-}
export RUNPOD_SKIP_TERMINATE=1
export EVA_RNA_TYPE=mRNA
export EVA_CHECKPOINT_DIR=~/eva_checkpoint/EVA_1.4B_CLM
export STORAGE_TARGET=local
export EVA_SMOKE_SINGLE_HOST=1
export EVA_NUM_SAMPLES=512
export EVA_CHUNK_SIZE=128
export EVA_OUT=data/processed/de_novo/generated.fasta
rm -f data/processed/de_novo/generated.fasta \
      data/processed/de_novo/generation_manifest.jsonl \
      data/processed/de_novo/raw_generated.fasta \
      data/processed/de_novo/run_state.json
rm -rf data/processed/de_novo/raw_chunks
python src/de_novo_hallucinations/eva_generate.py --smoke --resume \
  --chunk-size 128 \
  --output-fasta "$EVA_OUT" \
  --skip-s3-probe \
  --temperature 0.9
```

Expect after each chunk: `raw persisted: …` then `soft_filter: kept=…` then growing `generated.fasta`.

When AWS keys work later, re-run or `aws s3 sync` under `llm-batch/eva/v1/pilot512/`.

---

## Step 3 — Mac triage (after FASTA is local)

**Pull smoke and/or pilot from Macleod (Mac):**

```bash
mkdir -p data/processed/eva_pilot/de_novo
rsync -avz -e 'ssh -p 1024' \
  t41am25@127.0.0.1:~/Thermoswitches-MLBIO/data/processed/de_novo/ \
  /Users/amierzuhri/arcblanc/Thermoswitches-MLBIO/data/processed/eva_pilot/de_novo/
# also pull archived smoke if desired:
# rsync …:~/Thermoswitches-MLBIO/data/processed/eva_smoke32/ …/data/processed/eva_smoke32/
```

**Dynamic features + novelty + yield** (on Mac, after pilot FASTA exists):

```bash
cd /Users/amierzuhri/arcblanc/Thermoswitches-MLBIO
PYTHONPATH=src python src/thermo_sim/extract_fasta_dynamic_features.py \
  --fasta data/processed/eva_pilot/de_novo/generated.fasta \
  --output data/processed/eva_pilot/dynamic_features.csv \
  --workers 4

SOURCE_FASTA=data/processed/eva_pilot/de_novo/generated.fasta \
  bash scripts/run_eva_pilot_novelty.sh

PYTHONPATH=src python scripts/eva_yield_ratio.py \
  --dynamic-csv data/processed/eva_pilot/dynamic_features.csv
```

Yield: \(Z\le -2\) AND \(\Delta P_{\mathrm{RBS}}>0\) AND \(E_{\mathrm{Rfam}}>10^{-3}\) (no hit ⇒ \(E=\infty\)).

---

## Session: Step 2 complete + Mac pull (2026-08-11 23:18 → 2026-08-12 03:16)

Soft-drop re-run on `gpu02` **succeeded**.

| Item | Result |
|------|--------|
| Target | 512 E. coli (`EVA_SMOKE_SINGLE_HOST=1`, `T=0.9`, chunk 128) |
| Accepted | **512/512** in `generated.fasta` (`eva_sample_0`…`511`) |
| Soft-dropped | **6** across the run (not a hard abort) |
| Last chunk | `--num_seqs 7` top-up; `soft_filter: kept=7/7` |
| Wall | ~4 h (23:18–03:16); checkpoints at 253, 380, 512 |
| GPU | `exit` from `srun` after `grep -c '^>'` = 512 |

Mac pull (F5 tunnel, `ssh -p 1024`):

```text
data/processed/eva_pilot/de_novo/
  generated.fasta
  generation_manifest.jsonl
  raw_generated.fasta
  raw_chunks/ecoli_idx{0,126,253,380}_n128.fasta
  raw_chunks/ecoli_idx506_n7.fasta
```

### Step 3 complete (Mac CPU) — Path A

Vienna dynamics **done** (~03:20–04:00, ~41 min): 512 rows → `data/processed/eva_pilot/dynamic_features.csv`.

Novelty **done** (~04:01–04:57, ~1.9 h): blastn + nhmmer vs Rfam 14.9.

| Novelty category (E≤0.1 search) | Count | Fraction |
|---------------------------------|------:|---------:|
| identical / near (≥90% id) | 107 | 20.9% |
| remote homolog | 123 | 24.0% |
| **no hit** | **282** | **55.1%** |

Yield join (`eva_yield_ratio.py`) — gates \(Z\le-2\), \(\Delta P_{\mathrm{RBS}}>0\), \(E_{\mathrm{Rfam}}>10^{-3}\) (no hit ⇒ \(E=\infty\)):

| Metric | Value |
|--------|------:|
| 1. N quality-gated | **512** |
| 2. Structurally ordered (\(Z\le-2\)) | **58** (11.3%) |
| 3. Thermal RBS open (\(\Delta P_{\mathrm{RBS}}>0\)) | **457** (89.3%) |
| 4. Novel (\(E_{\mathrm{Rfam}}>10^{-3}\) or no hit) | **351** (of which 282 no-hit/inf) |
| 5. **Biophysical yield** (all three) | **31 / 512 = 6.05%** |

Artifacts:

- `data/processed/eva_pilot/yield_ratio.json`
- `data/processed/eva_pilot/yield_ratio_sequences.csv`
- `data/processed/eva_pilot/top_candidates.fasta` (**31** passers)
- `data/processed/novelty/eva_pilot_novelty_{report,summary,by_category}.*`

### Decision — **Path A (successful yield)**

Yield **6.05% ≫ 0.5–2%** band; **31** sequences pass all three gates. Not GenerRNA-style total memorization (55% no Rfam hit at E≤0.1). Unguided EVA at T=0.9 already emits a usable density of novel, switch-like candidates.

**Action (do next, not automatic):**

1. Keep `top_candidates.fasta` as the pilot panel (done).
2. Optionally email Dr. Angel with the five metrics above.
3. Only then green-light a **10k** Macleod drip (soft-drop, chunked) — do **not** start 10k until you consciously decide; pilot already proves the pipeline.

### Next right now

1. **Done (2026-08-13):** `eva2k` reached 2000/2000; Mac `--once --flush-remainder` finished slice_006. See [2k complete](#session-2k-complete--stream-triage-finished-2026-08-13).
2. Rank Top ~100 from pilot 31 + stream 74 (`eva_pilot/top_candidates.fasta` + `eva_stream/top_candidates.fasta`) by Z / ΔP_RBS.
3. 10k remains optional — do not start until you consciously decide.

---

## Session: 2k top-up + stream triage tooling (2026-08-12)

- Top-up: `EVA_NUM_SAMPLES=2000 --resume` → started at **remaining=1488/2000**; soft-drop intact.
- Observed ~17:55: **Checkpoint 896/2000**, next CLI 128 running.
- Mac tooling added (does not touch live job):
  - `scripts/eva_stream_triage.py` — `--source ssh|s3`, stride 250, poll 300s, complete-record FASTA parse, seed 512
  - `scripts/run_eva_slice_novelty.sh` — slice-scoped novelty paths
  - `eva_yield_ratio.py` — `--chunk-id` / rolling fields
  - `eva_generate.py` — writes `accepted_chunks/accepted_XXXX_YYYY.fasta` (deploy to cluster **after** this 2k finishes)
  - Seeded `data/processed/eva_stream/triage_state.json` at `last_triaged_count: 512`
- Post-slice analyze/log hooked in `eva_stream_triage.py` (`slice_*_analysis.json`, `stream_triage_log.md`, `top_candidates.fasta`, append to this file). Backfill: `--backfill-analysis`.
- **~19:22:** `slice_001` (512–761) complete — yield **12/250 = 4.80%** (−1.25 pp vs pilot). Watermark 762. F5 tunnel then dropped (`:1024` refused); watcher held open via failure-counter reset. Reconnect F5 + restart watcher to pick up analysis hook for later slices.

## Stream triage results

### slice_001 triage (eva_sample_512 … eva_sample_761) — 2026-08-12T18:47:10Z

| Metric | Value |
|--------|-------|
| Slice yield | **12/250 = 4.80%** |
| vs pilot 6.05% | -1.25 pp |
| Z≤−2 | 23/250 |
| ΔP_RBS>0 | 223/250 |
| E_Rfam>1e−3 | 168/250 |
| Rolling yield | 12/250 = 4.80% |

Novelty (E≤0.1):

| Category | n | % |
|----------|---|---|
| identical_near | 61 | 24.4% |
| remote_homolog | 71 | 28.4% |
| no_hit | 118 | 47.2% |

Top passers (by Z, then ΔP_RBS):

| ID | Z | ΔP_RBS | E_Rfam |
|----|---|--------|--------|
| eva_sample_523 | -7.019 | 0.0078 | inf |
| eva_sample_606 | -3.007 | 0.0091 | 0.025 |
| eva_sample_731 | -2.972 | 0.0215 | 0.087 |
| eva_sample_739 | -2.812 | 0.1196 | 0.055 |
| eva_sample_650 | -2.752 | 0.0215 | inf |
| eva_sample_720 | -2.679 | 0.0244 | inf |
| eva_sample_754 | -2.477 | 0.0528 | inf |
| eva_sample_648 | -2.455 | 0.0002 | 0.087 |
| eva_sample_691 | -2.416 | 0.0588 | inf |
| eva_sample_701 | -2.166 | 0.0207 | inf |

Artifacts: `/Users/amierzuhri/arcblanc/Thermoswitches-MLBIO/data/processed/eva_stream/slice_001_analysis.json`, `/Users/amierzuhri/arcblanc/Thermoswitches-MLBIO/data/processed/eva_stream/top_candidates.fasta` (12 cumulative passers).

### slice_002 triage (eva_sample_762 … eva_sample_1011) — 2026-08-12T22:50:25Z

| Metric | Value |
|--------|-------|
| Slice yield | **10/250 = 4.00%** |
| vs pilot 6.05% | -2.05 pp |
| Z≤−2 | 22/250 |
| ΔP_RBS>0 | 222/250 |
| E_Rfam>1e−3 | 173/250 |
| Rolling yield | 22/500 = 4.40% |

Novelty (E≤0.1):

| Category | n | % |
|----------|---|---|
| identical_near | 55 | 22.0% |
| remote_homolog | 61 | 24.4% |
| no_hit | 134 | 53.6% |

Top passers (by Z, then ΔP_RBS):

| ID | Z | ΔP_RBS | E_Rfam |
|----|---|--------|--------|
| eva_sample_813 | -12.386 | 0.0463 | inf |
| eva_sample_886 | -7.583 | 0.0315 | 0.021 |
| eva_sample_911 | -5.190 | 0.0405 | inf |
| eva_sample_961 | -4.542 | 0.0249 | inf |
| eva_sample_998 | -4.467 | 0.0514 | inf |
| eva_sample_997 | -3.491 | 0.1256 | inf |
| eva_sample_938 | -3.337 | 0.0642 | inf |
| eva_sample_882 | -2.325 | 0.0013 | 0.087 |
| eva_sample_861 | -2.162 | 0.0071 | inf |
| eva_sample_885 | -2.089 | 0.0244 | 0.017 |

Artifacts: `/Users/amierzuhri/arcblanc/Thermoswitches-MLBIO/data/processed/eva_stream/slice_002_analysis.json`, `/Users/amierzuhri/arcblanc/Thermoswitches-MLBIO/data/processed/eva_stream/top_candidates.fasta` (22 cumulative passers).

### slice_003 triage (eva_sample_1012 … eva_sample_1261) — 2026-08-13T00:08:29Z

| Metric | Value |
|--------|-------|
| Slice yield | **8/250 = 3.20%** |
| vs pilot 6.05% | -2.85 pp |
| Z≤−2 | 31/250 |
| ΔP_RBS>0 | 216/250 |
| E_Rfam>1e−3 | 176/250 |
| Rolling yield | 30/750 = 4.00% |

Novelty (E≤0.1):

| Category | n | % |
|----------|---|---|
| identical_near | 57 | 22.8% |
| remote_homolog | 61 | 24.4% |
| no_hit | 132 | 52.8% |

Top passers (by Z, then ΔP_RBS):

| ID | Z | ΔP_RBS | E_Rfam |
|----|---|--------|--------|
| eva_sample_1028 | -5.797 | 0.1470 | 0.025 |
| eva_sample_1184 | -4.204 | 0.0060 | inf |
| eva_sample_1077 | -3.566 | 0.0264 | inf |
| eva_sample_1200 | -3.323 | 0.0189 | inf |
| eva_sample_1137 | -3.116 | 0.0451 | inf |
| eva_sample_1066 | -2.328 | 0.0228 | 0.094 |
| eva_sample_1212 | -2.252 | 0.0191 | inf |
| eva_sample_1178 | -2.174 | 0.0395 | inf |

Artifacts: `/Users/amierzuhri/arcblanc/Thermoswitches-MLBIO/data/processed/eva_stream/slice_003_analysis.json`, `/Users/amierzuhri/arcblanc/Thermoswitches-MLBIO/data/processed/eva_stream/top_candidates.fasta` (30 cumulative passers).

### slice_004 triage (eva_sample_1262 … eva_sample_1511) — 2026-08-13T01:26:04Z

| Metric | Value |
|--------|-------|
| Slice yield | **12/250 = 4.80%** |
| vs pilot 6.05% | -1.25 pp |
| Z≤−2 | 26/250 |
| ΔP_RBS>0 | 226/250 |
| E_Rfam>1e−3 | 160/250 |
| Rolling yield | 42/1000 = 4.20% |

Novelty (E≤0.1):

| Category | n | % |
|----------|---|---|
| identical_near | 59 | 23.6% |
| remote_homolog | 64 | 25.6% |
| no_hit | 127 | 50.8% |

Top passers (by Z, then ΔP_RBS):

| ID | Z | ΔP_RBS | E_Rfam |
|----|---|--------|--------|
| eva_sample_1357 | -16.004 | 0.1589 | 0.011 |
| eva_sample_1379 | -5.357 | 0.0009 | 0.007 |
| eva_sample_1462 | -5.037 | 0.1145 | 0.0031 |
| eva_sample_1321 | -3.807 | 0.1146 | inf |
| eva_sample_1498 | -3.761 | 0.0145 | inf |
| eva_sample_1330 | -3.012 | 0.0381 | inf |
| eva_sample_1457 | -2.454 | 0.0732 | 0.087 |
| eva_sample_1316 | -2.440 | 0.0177 | inf |
| eva_sample_1505 | -2.374 | 0.0217 | 0.025 |
| eva_sample_1401 | -2.267 | 0.0630 | inf |

Artifacts: `/Users/amierzuhri/arcblanc/Thermoswitches-MLBIO/data/processed/eva_stream/slice_004_analysis.json`, `/Users/amierzuhri/arcblanc/Thermoswitches-MLBIO/data/processed/eva_stream/top_candidates.fasta` (42 cumulative passers).

### slice_005 triage (eva_sample_1512 … eva_sample_1761) — 2026-08-13T02:43:14Z

| Metric | Value |
|--------|-------|
| Slice yield | **18/250 = 7.20%** |
| vs pilot 6.05% | +1.15 pp |
| Z≤−2 | 35/250 |
| ΔP_RBS>0 | 224/250 |
| E_Rfam>1e−3 | 163/250 |
| Rolling yield | 60/1250 = 4.80% |

Novelty (E≤0.1):

| Category | n | % |
|----------|---|---|
| identical_near | 66 | 26.4% |
| remote_homolog | 57 | 22.8% |
| no_hit | 127 | 50.8% |

Top passers (by Z, then ΔP_RBS):

| ID | Z | ΔP_RBS | E_Rfam |
|----|---|--------|--------|
| eva_sample_1687 | -13.665 | 0.0534 | 0.0026 |
| eva_sample_1606 | -5.325 | 0.0936 | inf |
| eva_sample_1564 | -4.717 | 0.0721 | inf |
| eva_sample_1732 | -4.380 | 0.0237 | inf |
| eva_sample_1535 | -3.715 | 0.0602 | inf |
| eva_sample_1745 | -3.300 | 0.0128 | inf |
| eva_sample_1608 | -3.253 | 0.0416 | inf |
| eva_sample_1686 | -3.213 | 0.0526 | 0.087 |
| eva_sample_1545 | -2.873 | 0.0242 | 0.025 |
| eva_sample_1619 | -2.757 | 0.0027 | inf |

Artifacts: `/Users/amierzuhri/arcblanc/Thermoswitches-MLBIO/data/processed/eva_stream/slice_005_analysis.json`, `/Users/amierzuhri/arcblanc/Thermoswitches-MLBIO/data/processed/eva_stream/top_candidates.fasta` (60 cumulative passers).

### slice_006 triage (eva_sample_1762 … eva_sample_1999) — 2026-08-13T11:37:23Z

| Metric | Value |
|--------|-------|
| Slice yield | **14/238 = 5.88%** |
| vs pilot 6.05% | -0.17 pp |
| Z≤−2 | 29/238 |
| ΔP_RBS>0 | 216/238 |
| E_Rfam>1e−3 | 156/238 |
| Rolling yield | 74/1488 = 4.97% |

Novelty (E≤0.1):

| Category | n | % |
|----------|---|---|
| identical_near | 55 | 23.1% |
| remote_homolog | 61 | 25.6% |
| no_hit | 122 | 51.3% |

Top passers (by Z, then ΔP_RBS):

| ID | Z | ΔP_RBS | E_Rfam |
|----|---|--------|--------|
| eva_sample_1785 | -7.534 | 0.0581 | 0.077 |
| eva_sample_1998 | -4.353 | 0.0273 | inf |
| eva_sample_1858 | -3.750 | 0.0731 | inf |
| eva_sample_1826 | -3.670 | 0.0461 | inf |
| eva_sample_1808 | -2.808 | 0.0040 | inf |
| eva_sample_1914 | -2.642 | 0.0347 | inf |
| eva_sample_1886 | -2.626 | 0.0288 | inf |
| eva_sample_1996 | -2.564 | 0.0026 | inf |
| eva_sample_1954 | -2.442 | 0.0018 | inf |
| eva_sample_1876 | -2.362 | 0.1145 | 0.004 |

Artifacts: `/Users/amierzuhri/arcblanc/Thermoswitches-MLBIO/data/processed/eva_stream/slice_006_analysis.json`, `/Users/amierzuhri/arcblanc/Thermoswitches-MLBIO/data/processed/eva_stream/top_candidates.fasta` (74 cumulative passers).

---

## Session: 2k complete + stream triage finished (2026-08-13)

**Yes — generation and Mac triage are finished.** Watcher exited after `--once --flush-remainder`. Watermark `last_triaged_count: 2000` (`eva_sample_1999`).

| Block | n | Pass | Yield |
|-------|---|------|-------|
| Pilot (0–511) | 512 | 31 | **6.05%** |
| Stream slice_001 | 250 | 12 | 4.80% |
| Stream slice_002 | 250 | 10 | 4.00% |
| Stream slice_003 | 250 | 8 | 3.20% |
| Stream slice_004 | 250 | 12 | 4.80% |
| Stream slice_005 | 250 | 18 | 7.20% |
| Stream slice_006 (flush) | 238 | 14 | 5.88% |
| Stream 512–1999 | 1488 | 74 | **4.97%** |
| **All 2000** | **2000** | **105** | **5.25%** |

Still well above the 0.5–2% band. Stream novelty no-hit stayed ~47–54% (not GenerRNA-style collapse).

Artifacts:

- `data/processed/eva_stream/top_candidates.fasta` — 74 stream passers
- `data/processed/eva_pilot/top_candidates.fasta` — 31 pilot passers
- `data/processed/eva_stream/yield_ratio.json` / `stream_triage_log.md`

**Optional next:** merge + rank Top ~100 by Z then ΔP_RBS. Free the gpu02 job (`exit` the `srun` inside `eva2k`) if it is still holding the MIG. 10k is still a conscious decision, not automatic.
