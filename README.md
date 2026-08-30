# Thermoswitches-MLBIO

[![CI](https://github.com/arcblanc/Thermoswitches-MLBIO/actions/workflows/ci.yml/badge.svg)](https://github.com/arcblanc/Thermoswitches-MLBIO/actions/workflows/ci.yml)

**Author:** Amier Mohd Zuhri · MSc Data Science, University of Aberdeen (2026)  
**Thesis:** [*In Silico De Novo Design and Validation of Synthetic RNA Thermoswitches Via Deep Generative Modelling*](thesis/In_Silico_de_novo_Design_and_Validation_of_Synthetic_RNA_Thermoswitches.pdf)

Executable analysis stack for the thesis: Rfam/RefSeq corpus curation, ViennaRNA/NUPACK thermodynamics, the 92-column Random Forest baseline, EVA generation on Macleod, yield triage, and in silico biophysical validation.

**Thesis PDF:** [`thesis/In_Silico_de_novo_Design_and_Validation_of_Synthetic_RNA_Thermoswitches.pdf`](thesis/In_Silico_de_novo_Design_and_Validation_of_Synthetic_RNA_Thermoswitches.pdf) (annotated draft: [`thesis/In_Silico_de_novo_Design_and_Validation_of_Synthetic_RNA_Thermoswitches_AAmarkings.pdf`](thesis/In_Silico_de_novo_Design_and_Validation_of_Synthetic_RNA_Thermoswitches_AAmarkings.pdf))

---

## Scientific summary

Engineered *Escherichia coli* hosts carry a yield dilemma at 37 °C, and premature leaky expression selects against productive biomass before heat-shock induction. RNA thermoswitches (RNATs) sequester the ribosome binding site in a stem-loop lock at 37 °C and expose it near 42 °C, which offers a chemically free route to decouple growth from production. We curated a balanced benchmark of 1,198 Rfam thermoswitches and 1,198 length/GC-matched RefSeq 5′ UTR controls ($N = 2{,}396$), scored every transcript with dual folding engines, and trained a 200-tree Random Forest on a **92-column feature matrix** of static 37 °C physics, k-mers, and SD–AUG spacing while we withheld melting scalars for post-hoc ranking. StratifiedGroupKFold grouped on Rfam accession returned AUC ≈ **0.28**, which showed that static snapshot features memorised family-correlated trinucleotides rather than transferable switching rules. After that we deployed a frozen **EVA 1.4B CLM** on Macleod gpu02 with TaxID-conditioned `mRNA` prompts, drew 2,000 de novo 5′ UTR sequences, and ran automated biophysical triage in `scripts/triage/` to recover **105 yield-gated candidates** (5.25%) under

$$Z \le -2 \land \Delta P_{\mathrm{RBS}} > 0 \land E_{\mathrm{Rfam}} > 10^{-3}.$$

The yield-gated set advanced to continuous ViennaRNA thermal sweeps and Hill parameterisation in App 08, and the broader EVA cohort remained thermodynamically resistant across the 37–42 °C induction window with largely flat RBS exposure. This repository implements the three-domain architecture from the thesis: **supervised Random Forest baseline**, **EVA de novo generation**, and **in silico biophysical validation**.

---

## Pipeline overview

```
Rfam positives (1,198) ──┐
                         ├── CD-HIT 80% ── length/GC match ── Vienna + NUPACK ── 92-col X ── RF (200 trees)
RefSeq 5′ UTRs (1,198) ──┘                              │              │
                                                        │              └── OOF ŷ bins (post-hoc only)
                                                        └── Tm, Hill, Z, ΔP_RBS (withheld from X)

EVA 1.4B (frozen, TaxID-conditioned mRNA) ── Macleod gpu02 ── generated.fasta (2,000)
        │
Mac CPU ── Vienna Z / ΔP_RBS + Rfam novelty ── yield gate ── 105 candidates ── App 08 biophysical audit
```

| Stage | Role | Primary code |
|-------|------|----------------|
| **1. Corpus** | Rfam thermoswitch families vs RefSeq housekeeping 5′ UTRs | `src/data_engineering/` |
| **2. Features ($X$)** | Static 37 °C physics + composition + 16+64 k-mers + SD–AUG | `src/thermo_sim/noncircular_features.py` |
| **3. RF** | Bootstrap ensemble and grouped CV diagnostics | `src/thermo_sim/thermo_classifier.py` |
| **4. Post-hoc** | ŷ bins, ΔP_RBS, $n_H$, $T_m$, $Z$, Vienna–NUPACK Spearman | `src/thermo_sim/rf_posthoc.py` |
| **5. EVA** | Pretrained CLM generation (not trained here) | `src/de_novo_hallucinations/eva_generate.py` |
| **6. Triage** | Three-gate yield on streaming FASTA | `scripts/triage/` |
| **7. De novo audit** | EVA vs Rfam/RefSeq melting cohorts | `src/thermo_sim/eva_denovo_characterization.py` |

---

## Analysis map

| Content | Repository location | Regenerate / view |
|---------|---------------------|-------------------|
| **Panel construction** (matching, 92-column matrix, engines) | `src/data_engineering/`, `src/thermo_sim/` | [Rebuild core panel](#rebuild-core-panel) |
| **RF results** (grouped permutation, gates) | `data/processed/rf_*.json`, App 07 | `notebooks/07_noncircular_rf_results.py` |
| **Results figures** (ROC, permutation, funnel, Hill ribbons) | `notebooks/figures/thesis_figures/` | `notebooks/thesis_figures.py`, `src/thermo_sim/thesis_results_figures.py` |
| **Software stack and Macleod job** | `environment.yml`, `cluster/MACLEOD_EVA.md` | `cluster/EVA_OVERVIEW.md` |
| **92-col matrix and EVA soft-drop** | `noncircular_features.py`, `eva_quality.py` | `tests/test_noncircular_features.py`, `tests/test_eva_quality.py` |
| **RF formulae and yield gates** | `thermo_classifier.py`, `rf_posthoc.py` | `notebooks/rf_noncircular_methodology.md` |
| **105 yield table and attrition** | `data/processed/eva_denovo_checklist.json` | App 08, `notebooks/08_summarised_findings.md` |
| **GenerRNA prototype and lead-315 PoC** | `src/de_novo_hallucinations/genererna/`, App 08 figures | `notebooks/figures/08_eva_denovo/` |

---

## Repository layout

```
Thermoswitches-MLBIO/
├── thesis/                      # Thesis PDF (final + annotated draft)
├── src/                         # Importable library (uv editable install)
│   ├── data_engineering/        # Rfam extract, RefSeq UTRs, CD-HIT, length/GC match
│   ├── thermo_sim/              # Vienna/NUPACK, RF, post-hoc, batch, thesis figures
│   ├── de_novo_hallucinations/  # EVA + GenerRNA wrappers, quality gates
│   ├── novelty_eval/            # BLAST/nhmmer parsing, novelty reports
│   └── validation_embedding/    # BiRNA embedding (optional validation path)
├── scripts/                     # Runnable CLIs by domain → scripts/README.md
├── notebooks/                   # Marimo apps + exported figures
├── tests/                       # pytest (16 tests, no Vienna/NUPACK required)
├── cluster/                     # RunPod, EC2, Macleod EVA runbooks
├── data/                        # Raw reference + processed artefacts (.gitignore policy)
├── typings/                     # Stubs for optional NUPACK / ViennaRNA (ty)
├── .github/workflows/ci.yml     # ruff + ty + pytest
├── Makefile                     # make ci
├── pyproject.toml               # uv lockfile, dependency groups
└── environment.yml              # Conda bio-toolchain (cd-hit, infernal, hmmer)
```

**Split principle:** `src/` holds reusable logic, `scripts/` holds thin orchestration, and `notebooks/` are interactive viewers and audit apps (prefer loading JSON sidecars over retraining inline). Paths resolve via `src/data_engineering/paths.py` → `resolve_path()`.

---

## `src/` packages

| Package | Responsibility | Key modules |
|---------|----------------|-------------|
| **`data_engineering`** | Labelled corpus construction | `data_extraction.py`, `sequence_retrieval.py`, `cd_hit_sequence_similarity.py`, `refseq_utr_extract.py`, `cmscan_decontaminate.py`, `length_gc_match.py`, `knn_undersample.py` |
| **`thermo_sim`** | Folding, features, RF, evaluation | `vienna_rna.py`, `nupack_engine.py`, `thermo_batch.py`, `enrich_dynamic_features.py`, `noncircular_features.py`, `thermo_classifier.py`, `rf_posthoc.py`, `eva_denovo_characterization.py`, `thesis_results_figures.py` |
| **`de_novo_hallucinations`** | Sequence generation | `eva_generate.py`, `eva_quality.py`, `eva_prompts.py`, `genererna/` |
| **`novelty_eval`** | Homology search parsing | `parse_blast.py`, `parse_nhmmer.py`, `novelty_report.py` |
| **`validation_embedding`** | Optional embedding validation | `birna_embed.py`, `storage.py` |

---

## Feature matrix ($p = 92$)

We trained the Random Forest on 92 columns of static 37 °C physics, composition, k-mers, and SD–AUG spacing. Melting scalars (**$T_m$**, Hill, **$Z$**, **$\Delta P_{\mathrm{RBS}}$**) remained on the fused table for post-hoc ranking, and we withheld them from $X$.

**In $X$ (92 columns):**

- **7** static 37 °C biophysics (Vienna/NUPACK MFE per nt, ensemble diversity, mean positional entropy, max stem/loop length)
- **3** composition (%GC, length, $P_{\mathrm{paired,RBS}}(37\,^{\circ}\mathrm{C})$)
- **16 + 64** dinucleotide and trinucleotide frequencies (intensive, from matched FASTA)
- **2** SD–AUG spacing + missing-AUG sentinel (`sd_aug_spacing = -1`, `sd_aug_missing`)

Missing AUG was **not** dropped (RefSeq UTRs lack initiators more often than Rfam positives). Feature log: `data/processed/rf_noncircular_feature_log.json`.

---

## Key results

| Check | Result | Sidecar |
|-------|--------|---------|
| Length-alone AUC | ≈ 0.20 | `rf_noncircular_diagnostics.json` |
| **GroupKFold AUC (92-col $X$)** | **≈ 0.28** | same |
| EVA accepted / yield (2,000 panel) | 105 / 2,000 = **5.25%** | `eva_denovo_checklist.json` |
| Vienna vs NUPACK $T_m$ parity | Gates locked to **ViennaRNA only** | App 08, `notebooks/08_summarised_findings.md` |

---

## Marimo applications

```bash
uv run marimo edit notebooks/<app>.py
```

| App | File | Purpose |
|-----|------|---------|
| 1 | [`01_rfam_refseq_curation_eda.py`](notebooks/01_rfam_refseq_curation_eda.py) | Rfam vs RefSeq curation & EDA |
| 2 | [`02_prototype_benchmark.py`](notebooks/02_prototype_benchmark.py) | Vienna/NUPACK prototype benchmark |
| 3 | [`03_llm_smoke_test_results.py`](notebooks/03_llm_smoke_test_results.py) | GenerRNA smoke test |
| 4 | [`04_full_thermo_rf_analysis.py`](notebooks/04_full_thermo_rf_analysis.py) | Full thermo RF analysis (historical) |
| 5 | [`05_novelty_rfam_analysis.py`](notebooks/05_novelty_rfam_analysis.py) | Rfam novelty |
| 6 | [`06_classifier_architecture_ladder.py`](notebooks/06_classifier_architecture_ladder.py) | Feature-set architecture ablation (historical) |
| 7 | [`07_noncircular_rf_results.py`](notebooks/07_noncircular_rf_results.py) | **Results viewer** (loads JSON, no retrain) |
| 8 | [`08_eva_denovo_biophysical_characterization.py`](notebooks/08_eva_denovo_biophysical_characterization.py) | EVA vs Rfam/RefSeq melting audit |
| 9 | [`09_baseline_compositional_ablation.py`](notebooks/09_baseline_compositional_ablation.py) | Compositional ablation |
| — | [`thesis_figures.py`](notebooks/thesis_figures.py) | Export Chapter 4 figures → `notebooks/figures/thesis_figures/` |

Briefs: [`notebooks/08_summarised_findings.md`](notebooks/08_summarised_findings.md), [`notebooks/rf_noncircular_results.md`](notebooks/rf_noncircular_results.md).

---

## Quick start

```bash
git clone https://github.com/arcblanc/Thermoswitches-MLBIO.git
cd Thermoswitches-MLBIO

uv sync --group dev --group biophysics --group notebooks --group cloud --group llm
cp .env.example .env    # EMAIL, NCBI_API_KEY

make ci                 # ruff + format check + uv check + pytest (matches CI)
```

- **NUPACK 4.1:** manual wheel from `nupack-4.1.0.1/package/` (licensed, gitignored).
- **ViennaRNA:** `uv sync --group biophysics` (import as `RNA`).
- **Conda alternative:** `conda env create -f environment.yml` for cd-hit, infernal, hmmer.

### Development commands

| Command | Action |
|---------|--------|
| `make ci` | Full pipeline (sync + lint + format + typecheck + test) |
| `make test` | `pytest tests/ -v` |
| `make lint` | `ruff check src scripts notebooks tests` |
| `make typecheck` | `uv check` (ty) |

CLI entry points: [`scripts/README.md`](scripts/README.md). HPC/EVA runbooks: [`cluster/MACLEOD_EVA.md`](cluster/MACLEOD_EVA.md), [`cluster/EVA_OVERVIEW.md`](cluster/EVA_OVERVIEW.md).

---

## Rebuild core panel

```bash
# Positives
python src/data_engineering/data_extraction.py
python src/data_engineering/sequence_retrieval.py --all
python src/data_engineering/cd_hit_sequence_similarity.py --all

# Negatives
bash scripts/extraction/download_refseq_genomes.sh
PYTHONPATH=src python src/data_engineering/refseq_utr_extract.py
PYTHONPATH=src python src/data_engineering/cmscan_decontaminate.py
PYTHONPATH=src python src/data_engineering/length_gc_match.py \
  --all-positives --cds-truncate \
  --negatives-csv data/processed/refseq_utr/candidates_clean.csv \
  --negatives-fasta data/processed/refseq_utr/candidates_clean.fasta

# Thermo + enrich → fused_features_refseq_dynamic.csv
python src/thermo_sim/thermo_batch.py --run
PYTHONPATH=src python src/thermo_sim/enrich_dynamic_features.py --workers 4

# RF training + post-hoc JSON sidecars
python src/thermo_sim/thermo_classifier.py train \
  --fused-csv data/processed/fused_features_refseq_dynamic.csv \
  --model-path data/processed/models/rf_thermoswitch_noncircular.joblib
python src/thermo_sim/thermo_classifier.py posthoc \
  --fused-csv data/processed/fused_features_refseq_dynamic.csv
```

Groups for out-of-family CV: Rfam accession (positives), `REFSEQ:{assembly}` (negatives).

---

## Data artefact policy

Large regenerated tables (`fused_features_*.csv`, Vienna/NUPACK CSVs, `.joblib` models, FASTA pools) are **gitignored** under `data/processed/`. Lightweight **JSON metric sidecars** are allowlisted for diff-friendly audit (see `.gitignore`). Prototype golden panel: `data/processed/prototype/` (4-seq smoke fixture).

| Path | Contents |
|------|----------|
| `data/processed/fused_features_refseq_dynamic.csv` | Labelled panel (local, not committed) |
| `data/processed/rf_noncircular_diagnostics.json` | GroupKFold AUC + grouped permutation |
| `data/processed/rf_posthoc_report.json` | ŷ bins, gates, Spearman, checklist |
| `data/processed/eva_denovo_checklist.json` | Four-gate pass rates by cohort |
| `notebooks/figures/thesis_figures/` | Chapter 4 export PNGs |
| `notebooks/figures/08_eva_denovo/` | App 08 audit figures |

---

See [`CHANGELOG.md`](CHANGELOG.md) for release notes.
