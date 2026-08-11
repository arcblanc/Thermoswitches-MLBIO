# Thermoswitches-MLBIO

## RNA Thermoswitch Engineering: Machine Learning and Biophysics Pipeline

**Author:** Amier Zuhri  
**Role:** Biotech Machine Learning Engineer  
**Domain:** Synthetic Biology, Bioinformatics, Machine Learning, RNA

## Project Overview

This project computationally classifies and engineers **prokaryotic RNA thermoswitches** (RNA thermometers). These are highly structured *cis*-regulatory non-coding RNA elements, typically in the 5' UTR, that sequester the Shine-Dalgarno (SD) sequence at low temperature and expose it upon melting as temperature rises.

**Labelled corpus (current baseline):**
- **Positives** — Rfam thermoswitch families (CD-HIT–deduplicated).
- **Negatives** — housekeeping **5′ UTRs from NCBI RefSeq** complete genomes (Pseudomonadota + Bacillota), length/GC-matched to positives. Short Rfam “non-switch” fragments are **not** used as production negatives (they caused a severe length confound).

**Engineering goal:** Build a Random Forest classifier for non-leaky thermoswitches at a target temperature, with a longer-term objective of engineering synthetic switches that activate near 55°C with minimal low-temperature leakiness and a sharp Hill coefficient.

## Repository Architecture

The codebase is organized as domain subpackages under `src/`. Configuration is separated from code: secrets and runtime settings load from `.env` via `python-dotenv` and `os.environ`, while data paths resolve from the repository root through `data_engineering.paths.resolve_path()` rather than machine-specific absolute paths.

| Package | Module | Role |
|---------|--------|------|
| `data_engineering/` | `paths.py` | `PROJECT_ROOT` and `resolve_path()` for portable file I/O |
| `data_engineering/` | `data_extraction.py` | Rfam SQL extraction (**positives**; legacy Rfam negatives retired) |
| `data_engineering/` | `sequence_retrieval.py` | NCBI Entrez FASTA fetch for Rfam coordinates (`EMAIL`, `NCBI_API_KEY`) |
| `data_engineering/` | `cd_hit_sequence_similarity.py` | CD-HIT homology filtering (positives) |
| `data_engineering/` | `knn_undersample.py` | Legacy k-mer ENN + RUS balancing (pre–RefSeq path) |
| `data_engineering/` | `refseq_utr_extract.py` | **Production negatives:** housekeeping 5′ UTRs from RefSeq |
| `data_engineering/` | `cmscan_decontaminate.py` | Infernal `cmscan --cut_ga` scrub of thermoswitch/riboswitch hits |
| `data_engineering/` | `length_gc_match.py` | Pair each Rfam positive to a RefSeq UTR twin (Z-space / Hungarian) |
| `thermo_sim/` | `thermo_common.py` | Shared dataset loading, temperature grid, Hill fitting, dinuc shuffle |
| `thermo_sim/` | `vienna_rna.py` | ViennaRNA melting, unpaired-probability, and dynamic features |
| `thermo_sim/` | `nupack_engine.py` | NUPACK test-tube ensemble features |
| `thermo_sim/` | `feature_fusion.py` | Join `viennarna_*` and `nupack_*` feature blocks |
| `thermo_sim/` | `enrich_dynamic_features.py` | Enrich fused CSV with Z / ΔP_RBS / ΔΔG / Q / S |
| `thermo_sim/` | `thermo_prototype.py` | 4-sequence stress-test benchmark |
| `thermo_sim/` | `thermo_batch.py` | Batched full-dataset extraction with RAM logging |
| `thermo_sim/` | `thermo_classifier.py` | Random Forest train/predict (intensive + dynamic features) |
| `thermo_sim/` | `plot_prototype_benchmark.py` | Prototype benchmark figures |
| `de_novo_hallucinations/` | `eva_generate.py` / `eva_quality.py` / `eva_prompts.py` | EVA Option B panel generation + chunk quality gates |
| `de_novo_hallucinations/` | `gener_rna.py` | Legacy GenerRNA generation |
| `validation_embedding/` | `birna_embed.py` / `storage.py` | BiRNA-BERT embeddings + S3 / RunPod artifact sync |

### Configuration

Copy `.env.example` to `.env` and set NCBI credentials (used by Entrez fetch **and** RefSeq genome download):

```bash
# Boosts NCBI Datasets / Entrez rate limits (~3 req/s without a key → ~10/s with a key).
# scripts/download_refseq_genomes.sh sources .env and fails if NCBI_API_KEY is unset.
cat <<'EOF' > .env
EMAIL=your.email@university.edu
NCBI_API_KEY=your_ncbi_api_key_here
EOF
```

`sequence_retrieval.py` loads these at runtime with `load_dotenv()` and `os.environ.get()`. `download_refseq_genomes.sh` does `source .env` then `: "${NCBI_API_KEY:?...}"`. All pipeline scripts use repo-relative paths (for example `data/processed/fused_features_refseq_dynamic.csv`) resolved through `resolve_path()`, so the same code runs locally and on a remote VM without path edits.

---

## Phase 1: Dataset Construction (Rfam positives + RefSeq negatives)

### Why RefSeq controls

The first balanced set used Rfam non-switch / cis-regulatory fragments as negatives. Those sequences were systematically **shorter** than thermoswitch positives (~162 nt vs ~346 nt). Vienna/NUPACK raw MFE scaled with length (r ≈ −0.89), so the RF reached ~0.95 stratified AUC largely as a **length detector**. Production negatives now come from a different RNA resource — **NCBI RefSeq genomic 5′ UTRs** — so controls are long, structured, non-regulatory housekeeping regions from the same bacterial phyla as the positives.

### 1a. Positives (Rfam)

1. **Rfam SQL extraction** — prokaryotic heat-shock thermoswitch families (~2,960 before dedup).
2. **FASTA retrieval** — Biopython `Entrez` fetch on Rfam coordinates (`seq_start`, `seq_end`).
3. **CD-HIT** — deduplicate positives at 80% identity → **1,198** representatives.

### 1b. Negatives (RefSeq housekeeping 5′ UTRs)

1. **Genome download** — capped complete + reference assemblies for **Pseudomonadota** and **Bacillota** (`scripts/download_refseq_genomes.sh`, NCBI Datasets CLI). Requires `NCBI_API_KEY` in `.env` (see [Configuration](#configuration)).
2. **UTR extract** — operon-aware 5′ windows [200–600 nt] upstream of housekeeping CDS (`refseq_utr_extract.py`); CDS-proximal end retained.
3. **Decontamination** — Infernal `cmscan --cut_ga` against thermoswitch/riboswitch CMs (`cmscan_decontaminate.py`) → clean candidate pool.
4. **Length/GC match** — each Rfam positive is paired 1:1 to a RefSeq UTR in standardized (length, GC) space (`length_gc_match.py --all-positives --cds-truncate`): cKDTree top-K + Hungarian assignment, gates `|ΔL|≤40`, `|ΔGC|≤0.05`.

**3′ CDS-proximal truncation:** when a longer RefSeq UTR must match a positive’s length exactly, the matcher keeps the **3′ end** (`seq[-length:]`) and discards the **5′ distal** flank. That preserves the Shine-Dalgarno / translation-initiation region immediately upstream of the CDS so downstream `viennarna_delta_P_RBS` (last 30 nt) evaluates the authentic cellular RBS window.

**Group cross-validation schema:** to enforce out-of-distribution evaluation via `StratifiedGroupKFold`, all sequences share a single grouping column (`rfam_acc`):

- **Positives:** Rfam family accession (e.g. `RF00038`), so entire structural families are held out together.
- **RefSeq negatives:** `REFSEQ:{assembly_accession}` (e.g. `REFSEQ:GCF_000006945.2`), so all 5′ UTRs from one bacterial genome stay in the same fold.

`scripts/rf_length_bias_diagnostics.py` defaults to `--group-col rfam_acc` and needs no separate `assembly_accession` column.

**Primary training table:** `data/processed/fused_features_refseq_dynamic.csv` (1,198 pos + 1,198 RefSeq-matched neg after thermo fold + dynamic enrichment).

**Legacy (do not use for new RF claims):** `data/processed/balanced/balanced_dataset.{csv,fasta}` and `fused_features.csv` — Rfam-vs-Rfam ENN/RUS balance that embeds the length trap.

```bash
# Prerequisites: .env with EMAIL + NCBI_API_KEY (see Configuration above).
# download_refseq_genomes.sh sources .env automatically.

bash scripts/download_refseq_genomes.sh
PYTHONPATH=src python src/data_engineering/refseq_utr_extract.py
PYTHONPATH=src python src/data_engineering/cmscan_decontaminate.py
PYTHONPATH=src python src/data_engineering/length_gc_match.py \
  --all-positives --cds-truncate \
  --negatives-csv data/processed/refseq_utr/candidates_clean.csv \
  --negatives-fasta data/processed/refseq_utr/candidates_clean.fasta
```

See [Length-bias remediation](#length-bias-remediation-refseq-5-utr-baseline) for metrics and feature details.

---

## Phase 2: Biophysical Feature Engineering (Dual-Engine)

| Track | Tooling | Role |
|-------|---------|------|
| ViennaRNA | `RNA.md`, partition functions ([ViennaRNA docs](https://www.tbi.univie.ac.at/RNA/documentation.html)) | Melting curves and SD-window unpaired probabilities |
| NUPACK | `Model`, `Tube`, `tube_analysis` ([NUPACK 4.1](https://docs.nupack.org/4.1/)) | Complementary test-tube thermodynamic view |

Both tracks fit melting/exposure curves to a Hill sigmoid to extract **Tm**, **Hill coefficient**, and **amplitude**.

**Outputs (legacy Rfam-balanced melt):**
- `data/processed/viennarna/features.csv`
- `data/processed/nupack/features.csv`
- `data/processed/fused_features.csv`

**Outputs (RefSeq-matched production path):**
- `data/processed/viennarna/refseq_matched_features.csv` / `nupack/refseq_matched_features.csv`
- `data/processed/fused_features_refseq_matched.csv` → after enrich: `fused_features_refseq_dynamic.csv`

> **License note:** NUPACK requires a paid subscription per the [NUPACK 4 license](https://docs.nupack.org/4.1/). Install the local wheel from `nupack-4.1.0.1/package/` (gitignored) into your venv before running thermodynamic jobs.

### Prototype benchmark (4-sequence stress test)

| Role | Sequence | Purpose |
|------|----------|---------|
| Canonical positive | FourU (73 nt) | Sigmoidal Hill fitting on the SD window |
| Anomaly | cspA (425 nt) | Flat/inverted melting curve stress test |
| Short negative | Guanidine-II (47 nt) | Leakiness baseline |
| O(N³) stress | cspA (512 nt) | Peak RAM profiling |

```bash
pip install nupack-4.1.0.1/package/nupack-4.1.0.1-cp312-cp312-macosx_11_0_arm64.whl

python src/thermo_sim/thermo_prototype.py
python src/thermo_sim/thermo_prototype.py --run
python src/thermo_sim/plot_prototype_benchmark.py
```

### Batch extraction (full dataset)

```bash
python src/thermo_sim/thermo_batch.py --run --limit 10 --batch-size 2 --workers 2
python scripts/verify_batch_outputs.py 10
```

---

## Phase 3: Machine Learning Classification

- **Algorithms:** Random Forest (baseline) and monotonic **XGBoost** (`train-xgb`)
- **Default feature set (intensive):** length-normalized MFE (`*_MFE_per_nt`), stem/loop fractions, Tm / Hill / amplitude, plus Vienna dynamic columns (MFE Z-score, ΔP_RBS, ΔΔG, ensemble diversity, mean positional entropy)
- **Legacy feature set:** raw MFE + absolute stem/loop (`--legacy-features`) retained for comparison only
- **XGBoost monotone_constraints:** aligned to `PHYSICS_FEATURE_COLUMNS` via `MONOTONE_CONSTRAINTS_BY_FEATURE` in `thermo_classifier.py` (Z / MFE_per_nt / ΔΔG = −1; ΔP_RBS / Q / S / NUPACK amplitude & hill = +1; stem/loop fracs and other intensive cols = 0)
- **Inputs:** dual feature blocks (`viennarna_*`, `nupack_*`) joined on `(rfamseq_acc, seq_start, seq_end)`
- **Honest CV:** `StratifiedGroupKFold` by `rfam_acc` (positives) / `REFSEQ:{assembly}` (RefSeq negatives) via `scripts/rf_length_bias_diagnostics.py` and `scripts/xgb_monotonic_diagnostics.py`

```bash
# Train intensive RF on RefSeq-matched + dynamic fused features
python src/thermo_sim/thermo_classifier.py train \
  --fused-csv data/processed/fused_features_refseq_dynamic.csv \
  --model-path data/processed/models/rf_thermoswitch_refseq_dynamic.joblib

python scripts/rf_length_bias_diagnostics.py \
  --fused-csv data/processed/fused_features_refseq_dynamic.csv \
  --output-json data/processed/refseq_dynamic_rf_diagnostics.json

# Monotonic XGBoost (same panel) + GroupKFold vs RF baseline
python src/thermo_sim/thermo_classifier.py train-xgb \
  --fused-csv data/processed/fused_features_refseq_dynamic.csv \
  --model-path data/processed/models/xgb_thermoswitch_refseq_dynamic.joblib

PYTHONPATH=src python scripts/xgb_monotonic_diagnostics.py \
  --fused-csv data/processed/fused_features_refseq_dynamic.csv \
  --output-json data/processed/xgb_refseq_dynamic_diagnostics.json \
  --rf-baseline-json data/processed/refseq_dynamic_rf_diagnostics.json
```

On the RefSeq-dynamic panel (n=2396), unconstrained RF GroupKFold AUC ≈ **0.19** and monotonic XGB GroupKFold AUC ≈ **0.20** (Δ ≈ +0.01); length-alone gate still passes (~0.20).
---

## Length-bias remediation (RefSeq 5′ UTR baseline)

Summary of why the RefSeq control bank exists and what the rematched RF reports. Dataset construction steps live in [Phase 1](#phase-1-dataset-construction-rfam-positives--refseq-negatives).

### Problem
Legacy RF (~0.95 stratified AUC) with **Rfam negatives** was largely a length detector: positives ~346 nt vs Rfam negatives ~162 nt; raw MFE vs length r ≈ −0.89; length-alone AUC ≈ 0.94.

### Fix
Swap negatives to **RefSeq housekeeping 5′ UTRs**, match length/GC globally, train on intensive + dynamic Vienna features, evaluate with `StratifiedGroupKFold` (Rfam family / RefSeq assembly groups).

### Feature set on the rematched corpus
- Intensive: `*_MFE_per_nt`, stem/loop fractions; Tm / Hill / amplitude kept.
- Dynamic (`enrich_dynamic_features.py`): dinucleotide-shuffle MFE Z, ΔP_RBS (last 30 nt), ΔΔG, ensemble diversity Q, mean positional entropy S.

### Current honest metrics (n=2396, dynamic intensive set)
| Test | AUC |
|------|-----|
| Length-alone | ~0.20 (length shortcut removed) |
| Stratified intensive | ~0.80 (optimistic / leakage-prone) |
| **StratifiedGroupKFold** | ~0.19 (no transferable out-of-family detector) |

Mean MFE Z: positives ≈ −2.54 vs RefSeq controls ≈ −1.34 (composition-relative structure signal present; still fails group CV).

Narrative: [`notebooks/05_BUSINESS_BRIEF_thermo_rf_results.md`](notebooks/05_BUSINESS_BRIEF_thermo_rf_results.md). Release notes: [`CHANGELOG.md`](CHANGELOG.md).

```bash
# After Phase 1 matching + thermo fold of matched negatives:
PYTHONPATH=src python src/thermo_sim/enrich_dynamic_features.py --workers 4
```

---

## Phase 4: De Novo Design and Foundation Models

Code lives under `src/de_novo_hallucinations/` (GenerRNA + **EVA** generation) and `src/validation_embedding/` (BiRNA-BERT embeddings).

### EVA generation (smoke → Option B panel)

EVA replaces GenerRNA for new de novo runs. Conditioning uses the official CLI TaxID wrapper (`--rna_type mRNA` + `--taxid`) — never `sRNA`. Full run is a **3-host panel** (E. coli 562 / Salmonella 28901 / Lister 1639 → 10k sequences) in **512-seq chunks** with hard quality gates (Invalid Biological Formatting, Length Violations, Repetitive Text Collapse). Gate failure aborts and **stops the RunPod pod**.

```bash
# Dry-run
python scripts/eva_cloud_batch.py --smoke --dry-run

# On pod after keys + checkpoint (see cluster/EVA_RUNPOD.md):
bash cluster/runpod_eva_thermopod.sh smoke --yes
bash cluster/runpod_eva_thermopod.sh full --yes
```

Operator details and required API/AWS keys: [`cluster/EVA_RUNPOD.md`](cluster/EVA_RUNPOD.md). Artifacts land under `s3://…/llm-batch/eva/v1/` (separate from GenerRNA `llm-batch/v1`).

**Docker image bake:** from an M3, SSH into a temporary **Linux x86_64** VM and run `bash scripts/build_push_eva_docker.sh --push` to publish `arcblanc/eva-model:v1`. The script hard-fails on macOS (CUDA/`flash-attn` cannot be baked on Apple Silicon).

### GenerRNA memorization (key finding)

Early RF scoring used the **legacy Rfam-balanced** corpus (~2,396 sequences with short Rfam negatives). GenerRNA proposals scored by that RF and novelty-screened against Rfam 14.9 (`blastn` + `nhmmer`, ≥90% identity) showed **98 of 99** top hits were identical or near-identical to known sequences — memorized/regurgitated strands, not new RNA. Only **1** was a remote homolog; **0** had no hit. New RF work should train on the **RefSeq-matched** fused table instead; novelty screening remains mandatory for any generative output.

See `notebooks/06_novelty_rfam_analysis.ipynb` and `data/processed/novelty/novelty_summary.json` for the full breakdown.

### Local LLM smoke test (CPU)

Install optional LLM dependencies (separate from the thermodynamics stack):

```bash
pip install torch>=2.1   # local only — RunPod template includes CUDA-matched PyTorch
pip install -r requirements-llm.txt
cp .env.example .env   # optional HF_TOKEN
```

First GenerRNA run downloads ~3.6 GB of weights into `models/genererna/` (gitignored).

```bash
# Micro-generation: 2 sequences, temperature=1.0, top_k=250
python src/de_novo_hallucinations/gener_rna.py --num-samples 2 --temperature 1.0 --top-k 250

# BiRNA-BERT NUC embeddings + save .npy under data/processed/validation_embedding/smoke/
python src/validation_embedding/birna_embed.py \
  --input-fasta data/processed/de_novo/smoke/generated.fasta

# End-to-end
bash scripts/llm_smoke_test.sh
bash scripts/blank_slate_llm_test.sh
```

See `notebooks/04_llm_smoke_test_results.ipynb` for a walkthrough of the generated FASTA and embedding outputs.

The same scripts run on a cloud GPU without code changes when CUDA is available.

### RunPod + EC2 (AWS path)

**Architecture and finalized choices:** [`cluster/CLOUD_PIPELINE.md`](cluster/CLOUD_PIPELINE.md) (RunPod ↔ S3 ↔ EC2, scripts, and every production adjustment).

**Operator runbook:** [`cluster/README-aws.md`](cluster/README-aws.md).

**RunPod Thermopod** (1× A100 80GB): SSH in (no SCP), run GenerRNA + BiRNA, upload to `s3://thermo-s3-bucket`, terminate pod.

**EC2 `aws-thermo-ec2`** (`c7i-flex.large`, 2 workers): already running. Two separate thermo jobs:

1. **Train (current)** — RefSeq-matched panel → `fused_features_refseq_dynamic.csv` → intensive/dynamic RF (Rfam positives + RefSeq UTR negatives)
2. **Train (legacy)** — Rfam-balanced `fused_features.csv` — length-confounded; keep for comparison only
3. **Predict** — 10k GenerRNA/EVA FASTA → predictions → novelty filter on top hits

```bash
# Mac → RunPod
bash scripts/runpod_ssh.sh

# Mac → EC2 thermo (remote)
bash scripts/thermo_ec2_run.sh train --run
aws s3 cp s3://thermo-s3-bucket/llm-batch/v1/de_novo/generated.fasta data/processed/de_novo/
bash scripts/scp_ec2.sh push-fasta
bash scripts/thermo_ec2_run.sh predict --run
bash scripts/scp_ec2.sh pull-predictions
```

### Roadmap

- **Classifier:** address GroupKFold collapse (~0.19 AUC) — sequence+physics models and/or monotonic constraints; do not treat legacy 0.95 AUC as a wet-lab gate
- **De novo design:** EVA Option B panel (mRNA + TaxID hosts) with chunk quality gates; address GenerRNA memorization on the legacy path before inverse folding / 55°C targeting
- **Validation and embeddings:** fine-tuning BiRNA-BERT / RiNALMo on the curated dataset
- **Structural context:** complementary structure representations where they improve design fidelity

---

## Setup and Installation

```bash
git clone https://github.com/arcblanc/Thermoswitches-MLBIO.git
cd Thermoswitches-MLBIO

conda env create -f environment.yml
conda activate thermoswitches-mlbio

# Or pip venv (install cd-hit and viennarna separately)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env   # must include EMAIL + NCBI_API_KEY before RefSeq download

# Positives from Rfam (legacy ENN/RUS balance optional; not the production negative bank)
python src/data_engineering/data_extraction.py
python src/data_engineering/sequence_retrieval.py --all
python src/data_engineering/cd_hit_sequence_similarity.py --all

# Production negatives: RefSeq housekeeping 5′ UTRs → match → fold → enrich
# download_refseq_genomes.sh sources .env and aborts if NCBI_API_KEY is missing
bash scripts/download_refseq_genomes.sh
PYTHONPATH=src python src/data_engineering/refseq_utr_extract.py
PYTHONPATH=src python src/data_engineering/cmscan_decontaminate.py
PYTHONPATH=src python src/data_engineering/length_gc_match.py \
  --all-positives --cds-truncate \
  --negatives-csv data/processed/refseq_utr/candidates_clean.csv \
  --negatives-fasta data/processed/refseq_utr/candidates_clean.fasta

python src/thermo_sim/vienna_rna.py --dry-run
python src/thermo_sim/nupack_engine.py --dry-run
```

**Phase 1–2 dependencies:**
- `viennarna` — bioconda or `pip install viennarna`
- `nupack` — local wheel from `nupack-4.1.0.1/package/`
- `infernal` + `ncbi-datasets-cli` — via `environment.yml` (RefSeq UTR path)
- `NCBI_API_KEY` in `.env` — required for `download_refseq_genomes.sh` (raises NCBI rate limit ~3/s → ~10/s)
