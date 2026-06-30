# GCP Spot VM — GenerRNA + BiRNA-BERT batch

Run a walk-away Spot GPU job that generates sequences, computes NUC embeddings, syncs to GCS, and deletes the VM on success.

**Estimated Spot cost:** ~$2.76/hr for `a2-ultragpu-1g` (A100 80GB) in `us-central1` — verify current pricing in the GCP console.

## Two-phase rollout

1. **GPU smoke (2 sequences)** — validate CUDA + GCS wiring before spending on 10k:
   - Set instance metadata `generna-num-samples=2`
   - Expect ~1 hr wall time (mostly GenerRNA checkpoint download)
2. **Full batch (10,000 sequences)** — set `generna-num-samples=10000` (default)

Spot preemption is safe: manifests and FASTA are append-only and synced to GCS after each batch. Re-launch the same `gcloud` command to resume.

---

## 1. One-time setup (from your Mac)

```bash
gcloud services enable compute.googleapis.com storage.googleapis.com

export PROJECT_ID=$(gcloud config get-value project)
export GCS_BUCKET="${PROJECT_ID}-thermoswitches-mlbio"
export ZONE=us-central1-a

gsutil mb -l us-central1 -b on "gs://${GCS_BUCKET}"

# Upload startup script (re-upload after repo changes)
gsutil cp cluster/startup_spot_vm.sh "gs://${GCS_BUCKET}/cluster/startup_spot_vm.sh"
```

Grant the default compute service account `roles/storage.objectAdmin` on the bucket, and `roles/compute.instanceAdmin.v1` on the project if you want the VM to self-delete.

Check GPU quota:

```bash
gcloud compute regions describe "${ZONE%-*}" --format="value(quotas)"
```

---

## 2. Launch Spot VM

**GPU smoke (2 sequences):**

```bash
gcloud compute instances create rna-generative-spot-vm \
  --project="${PROJECT_ID}" \
  --zone="${ZONE}" \
  --machine-type=a2-ultragpu-1g \
  --provisioning-model=SPOT \
  --instance-termination-action=DELETE \
  --image-family=common-cu124-debian-12-gpu \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=200GB \
  --maintenance-policy=TERMINATE \
  --scopes=storage-full,cloud-platform \
  --metadata=gcs-bucket="${GCS_BUCKET}",generna-num-samples=2,startup-script-url="gs://${GCS_BUCKET}/cluster/startup_spot_vm.sh"
```

**Full batch (10,000 sequences):**

```bash
gcloud compute instances create rna-generative-spot-vm \
  --project="${PROJECT_ID}" \
  --zone="${ZONE}" \
  --machine-type=a2-ultragpu-1g \
  --provisioning-model=SPOT \
  --instance-termination-action=DELETE \
  --image-family=common-cu124-debian-12-gpu \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=200GB \
  --maintenance-policy=TERMINATE \
  --scopes=storage-full,cloud-platform \
  --metadata=gcs-bucket="${GCS_BUCKET}",generna-num-samples=10000,hf-token=YOUR_HF_TOKEN,startup-script-url="gs://${GCS_BUCKET}/cluster/startup_spot_vm.sh"
```

Optional metadata keys:

| Key | Purpose |
|-----|---------|
| `gcs-bucket` | GCS bucket for artifacts |
| `generna-num-samples` | `2` (smoke) or `10000` (batch) |
| `hf-token` | Hugging Face token (faster downloads) |
| `repo-url` | Override git clone URL (default: main repo) |

---

## 3. Monitor

```bash
gcloud compute instances get-serial-port-output rna-generative-spot-vm --zone="${ZONE}"
gsutil ls "gs://${GCS_BUCKET}/llm-batch/v1/"
```

Startup log on the VM: `/var/log/thermoswitches-llm-startup.log`

---

## 4. GCS artifact layout

```
gs://{bucket}/llm-batch/v1/
  de_novo/generated.fasta
  de_novo/generation_manifest.jsonl
  validation_embedding/manifest.jsonl
  validation_embedding/sample_*.npy
  validation_embedding/sample_*.json
  run_state.json
```

---

## 5. Recovery after Spot preemption

Re-run the same `gcloud compute instances create ...` command. The startup script clones fresh, sets `STORAGE_TARGET=gcs`, and `llm_cloud_batch.py` calls `sync_down` + `--resume` to continue from manifests.

---

## 6. Local dry-run (no GCS)

```bash
python scripts/llm_cloud_batch.py --dry-run
```
