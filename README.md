# Thermoswitches-MLBIO

## RNA Thermoswitch Engineering: Machine Learning and Biophysics Pipeline

**Author:** Amier Zuhri  
**Role:** Biotech Machine Learning Engineer  
**Domain:** Synthetic Biology, Bioinformatics, Machine Learning, RNA

## Project Overview

This project computationally classifies and engineers **prokaryotic RNA thermoswitches** (RNA thermometers). These are highly structured *cis*-regulatory non-coding RNA elements, typically in the 5' UTR, that sequester the Shine-Dalgarno (SD) sequence at low temperature and expose it upon melting as temperature rises.

**Engineering goal:** Build a Random Forest classifier for non-leaky thermoswitches at a target temperature, with a longer-term objective of engineering synthetic switches that activate near 55°C with minimal low-temperature leakiness and a sharp Hill coefficient.

## Repository Architecture

The codebase is organized as domain subpackages under `src/`. Configuration is separated from code: secrets and runtime settings load from `.env` via `python-dotenv` and `os.environ`, while data paths resolve from the repository root through `data_engineering.paths.resolve_path()` rather than machine-specific absolute paths.

| Package | Module | Role |
|---------|--------|------|
| `data_engineering/` | `paths.py` | `PROJECT_ROOT` and `resolve_path()` for portable file I/O |
| `data_engineering/` | `data_extraction.py` | Rfam SQL extraction (positives and negatives) |
| `data_engineering/` | `sequence_retrieval.py` | NCBI Entrez FASTA fetch (reads `EMAIL`, `NCBI_API_KEY` from `.env`) |
| `data_engineering/` | `cd_hit_sequence_similarity.py` | CD-HIT homology filtering |
| `data_engineering/` | `knn_undersample.py` | k-mer ENN + RUS balancing |
| `data_engineering/` | `refseq_utr_extract.py` | Housekeeping 5′ UTR negatives from RefSeq genomes |
| `data_engineering/` | `cmscan_decontaminate.py` | Infernal `cmscan --cut_ga` scrub of thermoswitch/riboswitch hits |
| `data_engineering/` | `length_gc_match.py` | Z-space / cKDTree / Hungarian length–GC matching |
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

Copy `.env.example` to `.env` and set:

```bash
EMAIL=your.email@university.edu
NCBI_API_KEY=your_ncbi_api_key_here
```

`sequence_retrieval.py` loads these at runtime with `load_dotenv()` and `os.environ.get()`. All pipeline scripts use repo-relative paths (for example `data/processed/balanced/balanced_dataset.csv`) resolved through `resolve_path()`, so the same code runs locally and on a remote VM without path edits.

---

## Phase 1: Data Extraction and Balancing

1. **Rfam SQL extraction** — positives (~2,960 prokaryotic heat-shock thermoswitches) and negatives (~168,000 bacterial 5' UTRs and cis-regulatory elements).
2. **FASTA retrieval** — Biopython `Entrez` fetch using Rfam coordinates (`seq_start`, `seq_end`).
3. **Homology filtering (CD-HIT)** — independent deduplication of positives and negatives at 80% identity to reduce leakage.
4. **K-mer ENN + RUS balancing** — Edited Nearest Neighbors on k-mer features, then Random Under-Sampling to a 1:1 class ratio.

Output: `data/processed/balanced/balanced_dataset.{csv,fasta}` (~2,396 sequences).

For the **length-controlled baseline**, short Rfam negatives are replaced by RefSeq housekeeping 5′ UTRs and globally length/GC-matched (see [Length-bias remediation](#length-bias-remediation-refseq-5-utr-baseline) below).

---

## Phase 2: Biophysical Feature Engineering (Dual-Engine)

| Track | Tooling | Role |
|-------|---------|------|
| ViennaRNA | `RNA.md`, partition functions ([ViennaRNA docs](https://www.tbi.univie.ac.at/RNA/documentation.html)) | Melting curves and SD-window unpaired probabilities |
| NUPACK | `Model`, `Tube`, `tube_analysis` ([NUPACK 4.1](https://docs.nupack.org/4.1/)) | Complementary test-tube thermodynamic view |

Both tracks fit melting/exposure curves to a Hill sigmoid to extract **Tm**, **Hill coefficient**, and **amplitude**.

**Outputs:**
- `data/processed/viennarna/features.csv`
- `data/processed/nupack/features.csv`
- `data/processed/fused_features.csv` (after fusion)

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

- **Algorithm:** Random Forest Classifier
- **Default feature set (intensive):** length-normalized MFE (`*_MFE_per_nt`), stem/loop fractions, Tm / Hill / amplitude, plus Vienna dynamic columns (MFE Z-score, ΔP_RBS, ΔΔG, ensemble diversity, mean positional entropy)
- **Legacy feature set:** raw MFE + absolute stem/loop (`--legacy-features`) retained for comparison only
- **Inputs:** dual feature blocks (`viennarna_*`, `nupack_*`) joined on `(rfamseq_acc, seq_start, seq_end)`
- **Honest CV:** `StratifiedGroupKFold` by `rfam_acc` (positives) / `REFSEQ:{assembly}` (RefSeq negatives) via `scripts/rf_length_bias_diagnostics.py`

```bash
# Train intensive RF on RefSeq-matched + dynamic fused features
python src/thermo_sim/thermo_classifier.py train \
  --fused-csv data/processed/fused_features_refseq_dynamic.csv \
  --model-path data/processed/models/rf_thermoswitch_refseq_dynamic.joblib

python scripts/rf_length_bias_diagnostics.py \
  --fused-csv data/processed/fused_features_refseq_dynamic.csv \
  --output-json data/processed/refseq_dynamic_rf_diagnostics.json
```

---

## Length-bias remediation (RefSeq 5′ UTR baseline)

The legacy RF (~0.95 stratified AUC) was largely a **length detector**: positives averaged ~346 nt vs short Rfam negatives ~162 nt, and raw MFE correlated with length (r ≈ −0.89; length-alone AUC ≈ 0.94).

### Dataset pivot
1. Decommission short Rfam non-switch fragments as negatives.
2. Extract housekeeping 5′ UTRs [200–600 nt] from RefSeq complete genomes (Pseudomonadota + Bacillota) via `scripts/download_refseq_genomes.sh` + `refseq_utr_extract.py`.
3. Scrub thermoswitch/riboswitch hits with Infernal `cmscan --cut_ga` (`cmscan_decontaminate.py`).
4. Globally match each CD-HIT positive to a RefSeq negative in Z(length, GC) space (`length_gc_match.py --all-positives --cds-truncate`): cKDTree top-K + Hungarian assignment, `|ΔL|≤40`, `|ΔGC|≤0.05`, CDS-proximal truncation for exact length.

### Feature and validation changes
- Intensive features only (no raw MFE in the production RF path).
- Dynamic Vienna enrichment (`enrich_dynamic_features.py`): dinucleotide-shuffle MFE Z, ΔP_RBS (last 30 nt), ΔΔG, ensemble diversity Q, mean positional entropy S.
- Group hold-out CV: family / assembly boundaries (not random stratified alone).

### Current honest metrics (n=2396, dynamic intensive set)
| Test | AUC |
|------|-----|
| Length-alone | ~0.20 (length shortcut removed) |
| Stratified intensive | ~0.80 (optimistic / leakage-prone) |
| **StratifiedGroupKFold** | ~0.19 (no transferable out-of-family detector) |

Mean MFE Z: positives ≈ −2.54 vs RefSeq controls ≈ −1.34 (composition-relative structure signal present; still fails group CV).

Narrative for stakeholders: [`notebooks/05_BUSINESS_BRIEF_thermo_rf_results.md`](notebooks/05_BUSINESS_BRIEF_thermo_rf_results.md). Release notes: [`CHANGELOG.md`](CHANGELOG.md).

```bash
# Download capped RefSeq assemblies (≥50) then extract + decontaminate UTRs
bash scripts/download_refseq_genomes.sh
PYTHONPATH=src python src/data_engineering/refseq_utr_extract.py
PYTHONPATH=src python src/data_engineering/cmscan_decontaminate.py

# Match all positives; fold negatives; enrich dynamics
PYTHONPATH=src python src/data_engineering/length_gc_match.py \
  --all-positives --cds-truncate \
  --negatives-csv data/processed/refseq_utr/candidates_clean.csv \
  --negatives-fasta data/processed/refseq_utr/candidates_clean.fasta
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

The Random Forest is trained **only** on the Rfam balanced corpus (~2,396 sequences). GenerRNA was then used to propose de novo candidates, which were scored by the RF and filtered for biophysical plausibility. A novelty screen of the top **99** RF-filtered candidates against Rfam 14.9 (`blastn` + `nhmmer`, ≥90% identity threshold) found that **98 of 99** were identical or near-identical to known sequences — i.e. **memorized training/regurgitated strands**, not genuinely new RNA. Only **1** sequence qualified as a remote homolog (<90% identity, E≤0.1); **0** had no hit.

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

1. **Train** — 2,396 Rfam balanced sequences → `fused_features.csv` → Random Forest (no hallucinated sequences in training)
2. **Predict** — 10k GenerRNA FASTA (SCP'd from Mac) → `denovo_predictions.csv` → novelty filter on top hits

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

cp .env.example .env

python src/data_engineering/data_extraction.py
python src/data_engineering/sequence_retrieval.py --all
python src/data_engineering/cd_hit_sequence_similarity.py --all
python src/data_engineering/knn_undersample.py

python src/thermo_sim/vienna_rna.py --dry-run
python src/thermo_sim/nupack_engine.py --dry-run
```

**Phase 2 dependencies:**
- `viennarna` — bioconda or `pip install viennarna`
- `nupack` — local wheel from `nupack-4.1.0.1/package/`
