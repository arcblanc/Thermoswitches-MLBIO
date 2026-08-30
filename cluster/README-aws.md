# RunPod + AWS EC2 Pipeline

Hybrid workflow: **RunPod** (GenerRNA *or* **EVA** + BiRNA → S3) then **EC2 `aws-thermo-ec2`** (ViennaRNA + NUPACK + Random Forest). EC2 is already running; thermo files transfer via **SCP**. RunPod supports **SSH only** — no SCP/SFTP.

**EVA smoke / full panel runbook:** [`EVA_RUNPOD.md`](EVA_RUNPOD.md) (Option B TaxIDs, 512-chunk gates, `llm-batch/eva/v1`).

**EVA Docker bake (M3 SSH → Linux VM → Hub):** on a temporary Linux x86_64 VM run `bash scripts/eva/build_push_eva_docker.sh --push` to publish `arcblanc/eva-model:v1`. Do not bake CUDA EVA images on Apple Silicon.

## Architecture

| Step | Machine | Action |
|------|---------|--------|
| 0 | Mac → RunPod SSH | Connect, clone repo, bootstrap, run LLM batch |
| 1a | RunPod (legacy) | GenerRNA 10k → BiRNA → S3 `llm-batch/v1` → terminate |
| 1b | RunPod (EVA) | EVA smoke then 10k Option B panel → BiRNA → S3 `llm-batch/eva/v1` → terminate on success or quality-gate fail |
| 2 | EC2 (SSH) | Thermo batch on 2,396 balanced sequences → train RF |
| 3 | Mac | Pull `generated.fasta` from S3, SCP to EC2 |
| 4 | EC2 (SSH) | Thermo batch on 10k de novo FASTA → RF predict |
| 5 | Mac | SCP results back from EC2 |

---

## 1. One-time setup

### S3 bucket (RunPod LLM artifacts only)

```bash
aws s3 mb s3://thermo-s3-bucket --region us-east-1
```

Create IAM credentials with `s3:PutObject`, `s3:GetObject`, `s3:ListBucket` on `thermo-s3-bucket`. Configure these **inside RunPod** (paste into `.env` — SCP to RunPod is not supported).

### `.env` on Mac

```bash
cp .env.example .env
# Set EC2_HOST, EC2_SSH_KEY, AWS credentials for Mac-side S3 pull
```

### RunPod SSH (Thermopod — no SCP)

```bash
ssh x0ggh3d7lmi9yn-64410a9d@ssh.runpod.io -i ~/.ssh/id_ed25519

# Or from repo:
bash scripts/cloud/runpod_ssh.sh
```

**Inside the pod** — clone repo and create `.env`:

```bash
git clone https://github.com/arcblanc/Thermoswitches-MLBIO.git
cd Thermoswitches-MLBIO
nano .env   # paste STORAGE_TARGET=runpod, AWS keys, HF_TOKEN, RUNPOD_API_KEY
```

Required pod `.env` keys:

```
STORAGE_TARGET=runpod
AWS_S3_BUCKET=thermo-s3-bucket
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
HF_TOKEN=...
RUNPOD_POD_ID=x0ggh3d7lmi9yn
RUNPOD_API_KEY=...
RUNPOD_AUTO_TERMINATE=true
GENERNA_NUM_SAMPLES=10000
```

### EC2 bootstrap (once, over SSH)

EC2 instance `aws-thermo-ec2` (`c7i-flex.large`, 2 vCPU) must already be running.

```bash
# From Mac — SCP NUPACK Linux wheel first (licensed)
scp -i ~/.ssh/aws-thermo-ec2.pem -r nupack-4.1.0.1/ ubuntu@<EC2_HOST>:~/Thermoswitches-MLBIO/

# SSH to EC2 and bootstrap
ssh -i ~/.ssh/aws-thermo-ec2.pem ubuntu@<EC2_HOST>
cd ~/Thermoswitches-MLBIO
bash cluster/ec2_bootstrap_thermo.sh
```

---

## 2. RunPod LLM batch

**Smoke test first** — set `GENERNA_NUM_SAMPLES=2` in pod `.env`.

```bash
# Inside RunPod SSH session
bash cluster/runpod_thermopod.sh
```

Or manually:

```bash
pip install -r requirements-llm.txt -r requirements-aws.txt
export STORAGE_TARGET=runpod
python scripts/generation/llm_cloud_batch.py --dry-run
python scripts/generation/llm_cloud_batch.py
```

Verify from Mac:

```bash
aws s3 ls s3://thermo-s3-bucket/llm-batch/v1/de_novo/
aws s3 ls s3://thermo-s3-bucket/llm-batch/v1/validation_embedding/ | head
```

Pod stops automatically when `RUNPOD_AUTO_TERMINATE=true` after **any** GenerRNA or BiRNA job (or after `llm_cloud_batch.py` finishes, including on failure). Use the API pod id (`x0ggh3d7lmi9yn`), not the SSH username.

---

## 3. EC2 thermo batch 1 — train Random Forest

Run on EC2 (separate SSH session). Memory-safe defaults: `--workers 2`, `--batch-size 1`.

```bash
# On EC2
cd ~/Thermoswitches-MLBIO && source .venv/bin/activate
python scripts/cloud/thermo_ec2_batch.py train --run
```

Or from Mac:

```bash
bash scripts/cloud/thermo_ec2_run.sh train --run
```

Outputs on EC2:

- `data/processed/fused_features.csv`
- `data/processed/models/rf_thermoswitch.joblib`

Pull results:

```bash
bash scripts/cloud/scp_ec2.sh pull-fused
bash scripts/cloud/scp_ec2.sh pull-all-results
```

---

## 4. EC2 thermo batch 2 — predict on 10k de novo

**Mac:** pull FASTA from S3, SCP to EC2:

```bash
aws s3 cp s3://thermo-s3-bucket/llm-batch/v1/de_novo/generated.fasta \
  data/processed/de_novo/generated.fasta
bash scripts/cloud/scp_ec2.sh push-fasta
```

**EC2** (second job — run in a fresh SSH session):

```bash
python scripts/cloud/thermo_ec2_batch.py predict --run
```

**Mac:** pull predictions:

```bash
bash scripts/cloud/scp_ec2.sh pull-predictions
bash scripts/cloud/scp_ec2.sh pull-denovo-fused
```

Output: `data/processed/denovo_predictions.csv` with `record_id`, `prob_positive`, `predicted_label`.

---

## S3 layout (RunPod LLM only)

```
s3://thermo-s3-bucket/llm-batch/v1/
  de_novo/generated.fasta
  de_novo/generation_manifest.jsonl
  validation_embedding/manifest.jsonl
  validation_embedding/sample_*.npy
  validation_embedding/sample_*.json
  run_state.json
```

Thermo outputs stay on EC2 until SCP'd back — not uploaded to S3.

---

## Pre-flight checklist

1. S3 bucket exists; AWS credentials on RunPod pod
2. RunPod SSH works: `bash scripts/cloud/runpod_ssh.sh`
3. EC2 running; SCP works; NUPACK wheel installed
4. Mac `.env`: `EC2_HOST`, `EC2_USER`, `EC2_SSH_KEY`, `EC2_REPO_PATH`
5. Smoke: 2-seq RunPod run; 4-seq EC2 dry-run before full batches
