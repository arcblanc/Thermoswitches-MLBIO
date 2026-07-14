# EVA on RunPod — smoke then full panel

Operator runbook for [GENTEL-lab/EVA](https://github.com/GENTEL-lab/EVA) generation + BiRNA-BERT validation.

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

Create / start the pod manually in the RunPod console (programmatic start is out of scope). Prefer the **official EVA Docker** image so `--taxid` → Greengenes lookup and MoE CUDA stack work.

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
