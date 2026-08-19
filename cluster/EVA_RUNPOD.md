# EVA on RunPod — smoke then full panel

Operator runbook for [GENTEL-lab/EVA](https://github.com/GENTEL-lab/EVA) generation + BiRNA-BERT validation.

## Machine roles

| Machine | Role |
|---------|------|
| **M3 Mac** | Local ViennaRNA / NUPACK. **SSH client only** for Docker bake — never `docker build` EVA CUDA images on the laptop. |
| **Temporary Linux x86_64 VM** | Clone EVA, bake `arcblanc/eva-model:v1`, `docker push` to Hub, then tear down. |
| **Docker Hub** | Warehouse for the baked image. |
| **RunPod** | Pulls `arcblanc/eva-model:v1` for smoke / full panel after you paste API + AWS keys. |

## Bake EVA image (Docker Hub) — Steps 1–2

Do this **before** the first EVA smoke. The M3 only orchestrates; a Linux amd64 VM does the bake and push.

**One-shot from the Mac** (starts `aws-thermo-ec2` if IAM allows, then builds/pushes):

```bash
# Optional, required for docker push:
export DOCKERHUB_USERNAME=arcblanc
export DOCKERHUB_TOKEN=…   # Hub access token with write

bash scripts/start_eva_bake_vm.sh
```

If IAM user `Arcblanc` cannot `ec2:DescribeInstances` / `StartInstances`, start instance `i-0123fbf60559bd082` in the AWS console (`eu-west-2`), put the public IP in `.env` as `EC2_HOST`, then re-run the script.

**Disk:** a 25 GB root volume is not enough (CUDA devel base + `flash-attn` compile OOMs the disk). Resize the root EBS volume to **≥80 GB** in the console, then on the VM:

```bash
sudo growpart /dev/nvme0n1 1
sudo resize2fs /dev/root
df -h /
```

### Step 1 — Bake on the Linux VM (layer cake)

```bash
# From M3: ssh into your temp Linux x86_64 VM, then:

# Option A: use this repo's helper (recommended)
git clone https://github.com/arcblanc/Thermoswitches-MLBIO.git   # or scp/rsync the script
cd Thermoswitches-MLBIO
export DOCKERHUB_IMAGE=arcblanc/eva-model:v1
export EVA_SRC_DIR=~/EVA
bash scripts/build_push_eva_docker.sh

# Option B: manual (same as EVA docs, retagged for Hub)
git clone https://github.com/GENTEL-Lab/EVA.git
cd EVA
docker build -f docker/Dockerfile -t arcblanc/eva-model:v1 .
```

The Dockerfile installs CUDA PyTorch, `flash-attn`, MegaBlocks/deps, and `eva-generate`. Expect a long build and multi‑GB disk. Checkpoint weights are **not** baked into the image.

`scripts/build_push_eva_docker.sh` **hard-fails on macOS** so the bake cannot be started by mistake on an M3.

### Step 2 — Push to Docker Hub (warehouse)

Still **on the VM**:

```bash
docker login
bash scripts/build_push_eva_docker.sh --push
# or: docker push arcblanc/eva-model:v1
```

Then in RunPod: Template → **Container Image** = `arcblanc/eva-model:v1` (GPU / A100-class). Continue with smoke below.

Env knobs (also in `.env.example`):

```bash
DOCKERHUB_IMAGE=arcblanc/eva-model:v1
EVA_SRC_DIR=~/EVA
```

## Prerequisites (keys — provide before first smoke)

Paste into the **pod** `.env` (and optionally Mac `.env` for monitoring):

| Variable | Purpose |
|----------|---------|
| `RUNPOD_API_KEY` | Stop pod via REST after success / quality-gate fail |
| `RUNPOD_POD_ID` | API pod id (not SSH username) |
| `RUNPOD_AUTO_TERMINATE=true` | Enable auto-stop |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` | S3 put/get/list |
| `AWS_S3_BUCKET` | e.g. `thermo-s3-bucket` |
| `AWS_S3_PREFIX=llm-batch/eva/v1` | Isolate from GenerRNA `llm-batch/v1` |
| `STORAGE_TARGET=runpod` | Enable S3 + stop hooks |
| `HF_TOKEN` | Optional, faster checkpoint download |
| `EVA_RNA_TYPE=mRNA` | **Never `sRNA`** |
| `EVA_CHECKPOINT_DIR=models/eva/checkpoint` | HF download target |

Create / start the pod manually in the RunPod console using custom image **`arcblanc/eva-model:v1`** (after Steps 1–2).

## Conditioning (Option B)

| Host | TaxID | Full quota |
|------|-------|------------|
| E. coli | 562 | 3334 |
| Salmonella enterica | 28901 | 3333 |
| Listeria monocytogenes | 1639 | 3333 |

CLI shape (EVA wrapper builds Greengenes):

```bash
eva-generate --checkpoint ./checkpoint --format clm \
  --rna_type mRNA --taxid 562 --num_seqs 512 --output /tmp/chunk.fa
```

## Quality gates (per 512-seq chunk)

Hard fail → abort + **stop pod**:

1. Invalid Biological Formatting (non-AUGC / empty)
2. Length Violations (outside `EVA_MIN_LEN`–`EVA_MAX_LEN`)
3. Repetitive Text Collapse (mono/dimer / near-duplicates)

Inspect failure: `aws s3 cp s3://$AWS_S3_BUCKET/llm-batch/eva/v1/run_state.json -`

## Smoke first

```bash
# Inside pod after git clone / pull and .env paste
huggingface-cli download GENTEL-Lab/EVA --local-dir models/eva/checkpoint
bash cluster/runpod_eva_thermopod.sh smoke --yes
```

Or: `bash scripts/eva_smoke_test.sh`

S3 layout:

```text
s3://thermo-s3-bucket/llm-batch/eva/v1/
  de_novo/generated.fasta
  de_novo/generation_manifest.jsonl
  validation_embedding/...
  run_state.json
```

Smoke uses local `.../smoke/` paths under `de_novo` / `validation_embedding`.

## Full 10k panel (after smoke passes)

```bash
# Restart pod in console if stopped; re-SSH; ensure .env still present
bash cluster/runpod_eva_thermopod.sh full --yes
```

Chunks of `EVA_CHUNK_SIZE=512` per host until quotas complete; BiRNA + verify; pod stops.

## Dry-run / S3 probe

```bash
export STORAGE_TARGET=runpod AWS_S3_PREFIX=llm-batch/eva/v1
python scripts/eva_cloud_batch.py --smoke --dry-run
python src/de_novo_hallucinations/eva_generate.py --smoke --dry-run
```

Live S3 probe runs at the start of non-dry `eva_generate` (writes `run_state` with `s3_probe`).
