# EVA install + smoke on Macleod (`gpu02`)

Architecture + tmux overview (why this stack): [`EVA_OVERVIEW.md`](EVA_OVERVIEW.md).  
Setup errors (RunPod bake → Macleod): [`EVA_SETUP_OVERVIEW.md`](EVA_SETUP_OVERVIEW.md) · [`EVA_SETUP_LOG.md`](EVA_SETUP_LOG.md).

Use after CUDA PyTorch works in conda env `torch_mig` (see [`macleod_log.md`](macleod_log.md)).

**Goal:** install GENTEL-Lab EVA, download checkpoint, run a tiny CLM generate on one `3g.20gb` MIG (~19.5 GiB).  
**Storage:** local disk first (`STORAGE_TARGET=local`). Wire S3 later.

Official EVA: [GENTEL-Lab/EVA](https://github.com/GENTEL-Lab/EVA) — min **16 GB VRAM** (you have ~19.5 GiB).

---

## 0. Interactive GPU job

From `macleod1`:

```bash
srun -p gpu -w gpu02 --gres=gpu:3g.20gb:1 --pty bash -i
```

**Host constraint:** `gpu02` is CentOS 7 (**glibc 2.17**). Current Anaconda/`conda-forge` Python 3.10+ builds need glibc ≥ 2.28 and will fail with `GLIBC_2.28 not found`. Prefer **module Python + venv** or **Singularity**, not a fresh conda env.

Inside the job (module path):

```bash
module load python/3.11.9 cuda/12.4.0
# optional: reuse a venv on sharedscratch
# source $HOME/sharedscratch/venvs/torch_mod/bin/activate
export CUDA_VISIBLE_DEVICES=0
export CUDA_MODULE_LOADING=LAZY
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```
---

## 1. One-shot helper (recommended)

On the GPU node, with this repo available:

```bash
# If repo not on cluster yet (from login or GPU node):
# git clone https://github.com/arcblanc/Thermoswitches-MLBIO.git
# cd Thermoswitches-MLBIO

bash scripts/eva/macleod_eva_install_smoke.sh
```

Phases inside the script:

1. CUDA check  
2. Clone/update `~/EVA` + `pip install -e .` (+ try `flash-attn`)  
3. HuggingFace checkpoint → `~/eva_checkpoint` (or `$EVA_CHECKPOINT_DIR`)  
4. Tiny `eva-generate` (default **4** mRNA seqs, TaxID 562)  
5. Optional: this repo’s `eva_generate.py --smoke` with `STORAGE_TARGET=local` (generation only if `--skip-birna`)

Env knobs:

| Variable | Default | Meaning |
|----------|---------|---------|
| `EVA_SRC_DIR` | `$HOME/EVA` | EVA git checkout |
| `EVA_CHECKPOINT_DIR` | `$HOME/eva_checkpoint` | HF weights |
| `EVA_SMOKE_NUM_SEQS` | `4` | Tiny generate size |
| `THERMO_REPO` | auto / `$HOME/Thermoswitches-MLBIO` | This repo |
| `HF_TOKEN` | unset | Optional HF auth for faster download |

---

## 2. Manual steps (same as the script)

```bash
# --- EVA package ---
export EVA_SRC_DIR="${EVA_SRC_DIR:-$HOME/EVA}"
if [[ ! -d "$EVA_SRC_DIR/.git" ]]; then
  git clone https://github.com/GENTEL-Lab/EVA.git "$EVA_SRC_DIR"
fi
cd "$EVA_SRC_DIR"
pip install -U pip
pip install -e .
# Full MoE inference usually needs these (can take a long time to build):
pip install flash-attn --no-build-isolation || echo "flash-attn failed — retry later"
eva-generate --help | head

# --- Checkpoint (multi-GB) ---
export EVA_CHECKPOINT_DIR="${EVA_CHECKPOINT_DIR:-$HOME/eva_checkpoint}"
pip install -q huggingface_hub
huggingface-cli download GENTEL-Lab/EVA --local-dir "$EVA_CHECKPOINT_DIR"

# --- Tiny generate ---
mkdir -p "$HOME/eva_smoke_out"
eva-generate \
  --checkpoint "$EVA_CHECKPOINT_DIR" \
  --format clm \
  --rna_type mRNA \
  --taxid 562 \
  --num_seqs "${EVA_SMOKE_NUM_SEQS:-4}" \
  --output "$HOME/eva_smoke_out/smoke_ecoli.fa"

wc -l "$HOME/eva_smoke_out/smoke_ecoli.fa"
head -n 20 "$HOME/eva_smoke_out/smoke_ecoli.fa"
```

Watch GPU while generating (second SSH + `srun` or `ssh` to node if allowed):

```bash
nvidia-smi
```

---

## 3. Repo orchestrator smoke (after tiny generate works)

```bash
cd "${THERMO_REPO:-$HOME/Thermoswitches-MLBIO}"
export STORAGE_TARGET=local
export EVA_RNA_TYPE=mRNA
export EVA_CHECKPOINT_DIR="${EVA_CHECKPOINT_DIR:-$HOME/eva_checkpoint/EVA_1.4B_CLM}"
export EVA_NUM_SAMPLES=16
export PYTHONPATH=src:${PYTHONPATH:-}
export RUNPOD_SKIP_TERMINATE=1   # no RunPod on Macleod

python src/de_novo_hallucinations/eva_generate.py --smoke --dry-run
python src/de_novo_hallucinations/eva_generate.py --smoke \
  --output-fasta data/processed/de_novo/generated.fasta
```

Full smoke (EVA + BiRNA) needs `requirements-llm.txt` + BiRNA model IDs from `.env` — do that only after generation smoke passes.

---

## 4. Gated smoke → scale pilot → S3 (triage ladder)

Use after `eva-generate` 4-seq smoke works. Prefer **module Python + `torch_mod` venv**.

### Env (S3)

```bash
cd "${THERMO_REPO:-$HOME/Thermoswitches-MLBIO}"
module load python/3.11.9 cuda/12.4.0 gcc/12.3.0 git
source $HOME/sharedscratch/venvs/torch_mod/bin/activate
export CUDA_VISIBLE_DEVICES=0 CUDA_MODULE_LOADING=LAZY
export PYTHONPATH=src:${PYTHONPATH:-}
export RUNPOD_SKIP_TERMINATE=1
export EVA_RNA_TYPE=mRNA
export EVA_CHECKPOINT_DIR="${EVA_CHECKPOINT_DIR:-$HOME/eva_checkpoint/EVA_1.4B_CLM}"
export STORAGE_TARGET=s3
export AWS_S3_BUCKET=thermo-s3-bucket
export AWS_S3_PREFIX=llm-batch/eva/v1   # Step 1 smoke; use …/pilot512 for Step 2
export AWS_ACCESS_KEY_ID=… AWS_SECRET_ACCESS_KEY=… AWS_REGION=eu-west-2
# Always write batch FASTA path so sync_up uploads it:
export EVA_OUT=data/processed/de_novo/generated.fasta
```

### Step 1 — Soft-drop smoke (16–64, multi-TaxID)

```bash
export AWS_S3_PREFIX=llm-batch/eva/v1/smoke
export EVA_NUM_SAMPLES=32
export EVA_CHUNK_SIZE=32
python src/de_novo_hallucinations/eva_generate.py --smoke --resume \
  --chunk-size "$EVA_CHUNK_SIZE" \
  --output-fasta "$EVA_OUT"
# Expect soft_filter kept=… lines + raw_generated.fasta + generation_complete
```

Orchestrator behaviour (all steps):

1. Persist each CLI chunk to `raw_generated.fasta` + `raw_chunks/` **before** filtering.
2. **Soft-drop** failing sequences (keep the rest); log drop reasons in `run_state.json`.
3. Append passers to `generated.fasta` / `generation_manifest.jsonl` and checkpoint.
4. Top-up until each host quota is met (`EVA_SOFT_TOPUP_BUFFER`, default `1.05`).

### Step 2 — Scale pilot (512, E. coli only, 128-seq chunks)

```bash
export AWS_S3_PREFIX=llm-batch/eva/v1/pilot512
export STORAGE_TARGET=local   # or s3 once keys work
export EVA_SMOKE_SINGLE_HOST=1
export EVA_NUM_SAMPLES=512
export EVA_CHUNK_SIZE=128
# Fresh pilot: clear prior aborted run artifacts (optional if --resume from empty)
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
# After each ~128-seq chunk (~1.5 h): soft_filter + append + checkpoint
# Parallel shell: nvidia-smi -l 5
```

### Step 3 — Local Mac triage (CPU)

```bash
aws s3 sync s3://thermo-s3-bucket/llm-batch/eva/v1/pilot512/ data/processed/eva_pilot/

PYTHONPATH=src python src/thermo_sim/extract_fasta_dynamic_features.py \
  --fasta data/processed/eva_pilot/de_novo/generated.fasta \
  --output data/processed/eva_pilot/dynamic_features.csv \
  --workers 4

SOURCE_FASTA=data/processed/eva_pilot/de_novo/generated.fasta \
  bash scripts/extraction/run_eva_pilot_novelty.sh

PYTHONPATH=src python scripts/triage/eva_yield_ratio.py \
  --dynamic-csv data/processed/eva_pilot/dynamic_features.csv
# Yield: Z<=-2 AND ΔP_RBS>0 AND E_Rfam>1e-3 (no hit ⇒ E=inf)
```

### Stream triage (Mac, while GPU 2k runs)

Do **not** attach tmux on `gpu02` — sessions live on **macleod1** (`tmux attach -t eva2k`).

**SSH fallback** (live 2k is `STORAGE_TARGET=local`):

```bash
# Mac tmux 1: keep F5 tunnel (127.0.0.1:1024) alive
ssh -p 1024 t41am25@127.0.0.1 'echo ok'

# Mac tmux 2: poll every 300s; seed 512 so first slice is 513–762
cd ~/…/Thermoswitches-MLBIO   # or local repo path
PYTHONPATH=src python scripts/triage/eva_stream_triage.py \
  --source ssh \
  --remote t41am25@127.0.0.1 \
  --ssh-port 1024 \
  --remote-fasta /home/t41am25/Thermoswitches-MLBIO/data/processed/de_novo/generated.fasta \
  --stride 250 \
  --seed-count 512 \
  --poll-seconds 300 \
  --workers 4
```

**S3-native** (after keys work + `STORAGE_TARGET=s3` on generator — no SSH):

```bash
PYTHONPATH=src python scripts/triage/eva_stream_triage.py \
  --source s3 \
  --s3-bucket thermo-s3-bucket \
  --s3-prefix llm-batch/eva/v1/pilot2k \
  --stride 250 --seed-count 512 --poll-seconds 300 --workers 4
```

State: `data/processed/eva_stream/triage_state.json`. Masters: `dynamic_features_master.csv`, rolling `yield_ratio.json`. When generation finishes and `<250` remain: add `--once --flush-remainder`.

**After each slice** the watcher also:

- writes `slice_XXX_analysis.json` (gates, novelty mix, top passers, vs pilot 6.05%)
- appends `stream_triage_log.md` + `analysis_log.jsonl`
- rebuilds cumulative `top_candidates.fasta` (ranked by Z, then ΔP_RBS)
- appends a section to `cluster/macleod_log.md`

Backfill if a slice finished before this hook existed:

```bash
PYTHONPATH=src python scripts/triage/eva_stream_triage.py --backfill-analysis
```

Use an **absolute** remote FASTA path (`/home/t41am25/...`); `~/` expands on the Mac and breaks rsync.

---

## 5. Failure cheat sheet

| Symptom | Likely fix |
|---------|------------|
| `Requested node configuration is not available` | Use `--gres=gpu:3g.20gb:1` and `-p gpu` |
| `eva-generate: command not found` | Activate `torch_mod`; `pip install -e ~/EVA` |
| CUDA OOM | Lower `--num_seqs` / `EVA_BATCH_SIZE`; one MIG only |
| `flash-attn` build fails | `module load cuda/12.4.0 gcc/12.3.0`; `--no-build-isolation` |
| Checkpoint missing | Use `…/EVA_1.4B_CLM` under HF download dir |
| Import hangs on CUDA | `export CUDA_VISIBLE_DEVICES=0` (MIG job) |
| S3 FASTA missing | Pass `--output-fasta data/processed/de_novo/generated.fasta` |
| Hard gate abort loses whole chunk | Soft-drop is default; check `raw_generated.fasta` + `soft_filter` logs |
| Soft-drop removes entire chunk ×3 | Mode collapse — raise `--temperature` or abort and inspect raw FASTA |

---

## 6. When to stop / resume

```bash
exit   # ends srun job, frees MIG
```

Home installs (`~/EVA`, `~/eva_checkpoint`, `torch_mod`) persist across jobs. Next session: `srun` → activate venv → `eva_generate.py --resume`.
