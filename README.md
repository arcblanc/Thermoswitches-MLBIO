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
| `thermo_sim/` | `thermo_common.py` | Shared dataset loading, temperature grid, Hill fitting |
| `thermo_sim/` | `vienna_rna.py` | ViennaRNA melting and unpaired-probability features |
| `thermo_sim/` | `nupack_engine.py` | NUPACK test-tube ensemble features |
| `thermo_sim/` | `feature_fusion.py` | Join `viennarna_*` and `nupack_*` feature blocks |
| `thermo_sim/` | `thermo_prototype.py` | 4-sequence stress-test benchmark |
| `thermo_sim/` | `thermo_batch.py` | Batched full-dataset extraction with RAM logging |
| `thermo_sim/` | `plot_prototype_benchmark.py` | Prototype benchmark figures |
| `de_novo_hallucinations/` | *(planned)* | De novo sequence design and inverse folding |
| `validation_embedding/` | *(planned)* | Foundation-model validation and embedding workflows |

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
- **Inputs:** dual feature blocks (`viennarna_*`, `nupack_*`) joined on `(rfamseq_acc, seq_start, seq_end)`
- **Output:** majority-vote classification of functional thermoswitches

---

## Phase 4: De Novo Design and Foundation Models

Code lives under `src/de_novo_hallucinations/` (GenerRNA generation) and `src/validation_embedding/` (BiRNA-BERT embeddings).

### Local LLM smoke test (CPU)

Install optional LLM dependencies (separate from the thermodynamics stack):

```bash
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

### Cloud Spot VM (10k batch + GCS)

See [`cluster/README.md`](cluster/README.md) for bucket setup, the `gcloud` launch command, and two-phase rollout:

1. **GPU smoke** — metadata `generna-num-samples=2`
2. **Full batch** — metadata `generna-num-samples=10000`

```bash
# On the VM (or locally with STORAGE_TARGET=gcs in .env)
bash scripts/llm_cloud_run.sh
python scripts/llm_cloud_batch.py --dry-run
```

### Roadmap

- **De novo design:** inverse folding and mutation proposals targeting 55°C activation with low leakiness
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
