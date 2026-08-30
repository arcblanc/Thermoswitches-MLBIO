# Thermoswitches-MLBIO

[![CI](https://github.com/arcblanc/Thermoswitches-MLBIO/actions/workflows/ci.yml/badge.svg)](https://github.com/arcblanc/Thermoswitches-MLBIO/actions/workflows/ci.yml)

**Author:** Amier Zuhri · MSc Data Science, University of Aberdeen (2026)  
**Thesis:** *In Silico De Novo Design and Validation of Synthetic RNA Thermoswitches Via Deep Generative Modelling*

Executable analysis stack for the thesis: corpus curation, ViennaRNA/NUPACK thermodynamics, the non-circular Random Forest panel, EVA generation orchestration, yield triage, and figure export.

**Thesis chapter prose, guideline evaluations, and the supplementary booklet** live in the sibling package [`../thesis_md_package/`](../thesis_md_package/) (see [Thesis markdown package](#thesis-markdown-package)).

---

## Scientific summary

Prokaryotic RNA thermoswitches in the 5′ UTR sequester the Shine–Dalgarno (RBS) sequence at low temperature and expose it as temperature rises. We built a length/GC-matched panel ($N = 2{,}396$), scored sequences with dual folding engines, trained a Random Forest on **non-circular 37 °C physics and k-mers**, and used **post-hoc melting gates** (not RF inputs) to rank phenotype. A frozen **EVA 1.4B CLM** on Macleod GPU generated 2,000 sequences; Mac-side triage recovered **105 yield-gated candidates** under

$$Z \le -2 \land \Delta P_{\mathrm{RBS}} > 0 \land E_{\mathrm{Rfam}} > 10^{-3}.$$

The Random Forest is a **ranking aid**, not a wet-lab acceptance gate. GroupKFold AUC on the non-circular matrix is ≈ **0.28** (no transferable out-of-family detector); the circular 20-column history inflated to ≈ **0.80** under stratified CV because melting scalars leaked the phenotype into $X$.

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
| **3. RF** | Bootstrap ensemble; grouped CV diagnostics | `src/thermo_sim/thermo_classifier.py` |
| **4. Post-hoc** | ŷ bins, ΔP_RBS, $n_H$, $T_m$, $Z$, Vienna–NUPACK Spearman | `src/thermo_sim/rf_posthoc.py` |
| **5. EVA** | Pretrained CLM generation (not trained here) | `src/de_novo_hallucinations/eva_generate.py` |
| **6. Triage** | Three-gate yield on streaming FASTA | `scripts/triage/` |
| **7. De novo audit** | EVA vs Rfam/RefSeq melting cohorts | `src/thermo_sim/eva_denovo_characterization.py` |

---

## Thesis and supplementary map

| Thesis / supplementary content | Repository location | Regenerate / view |
|--------------------------------|---------------------|-------------------|
| **Chapter 3 methods** (matching, 92-column matrix, engines) | `src/data_engineering/`, `src/thermo_sim/` | [Rebuild core panel](#rebuild-core-panel) |
| **Chapter 4 results** (RF, grouped permutation, gates) | `data/processed/rf_*.json`, App 07 | `notebooks/07_noncircular_rf_results.py` |
| **Chapter 4 figures** (ROC, permutation, funnel, Hill ribbons) | `notebooks/figures/thesis_figures/` | `notebooks/thesis_figures.py`, `src/thermo_sim/thesis_results_figures.py` |
| **Supplementary §1** (software stack, Macleod job) | `environment.yml`, `cluster/MACLEOD_EVA.md` | `cluster/EVA_OVERVIEW.md` |
| **Supplementary §2** (92-col matrix, EVA soft-drop) | `noncircular_features.py`, `eva_quality.py` | `tests/test_noncircular_features.py`, `tests/test_eva_quality.py` |
| **Supplementary §3** (RF formulae, yield gates) | `thermo_classifier.py`, `rf_posthoc.py` | `notebooks/rf_noncircular_methodology.md` |
| **Supplementary §4** (105 yield table, attrition, parity) | `data/processed/eva_denovo_checklist.json` | App 08, `notebooks/08_summarised_findings.md` |
| **Supplementary §5–6** (GenerRNA prototype, lead-315 PoC) | `src/de_novo_hallucinations/genererna/`, App 08 figures | `notebooks/figures/08_eva_denovo/` |
| **Guideline evaluations & accuracy audits** | `../thesis_md_package/guideline_eval/` | Not in this repo |
| **Supplementary booklet (PDF/DOCX source)** | `../thesis_md_package/supplementary/` | `supplementary/build_supplementary_figures.py` |

Pointer: [`notebooks/THESIS_MD_PACKAGE.md`](notebooks/THESIS_MD_PACKAGE.md).

---

## Repository layout

```
Thermoswitches-MLBIO/
├── src/                         # Importable library (uv editable install)
│   ├── data_engineering/        # Rfam extract, RefSeq UTRs, CD-HIT, length/GC match
│   ├── thermo_sim/              # Vienna/NUPACK, RF, post-hoc, batch, thesis figures
│   ├── de_novo_hallucinations/  # EVA + GenerRNA wrappers, quality gates
│   ├── novelty_eval/            # BLAST/nhmmer parsing, novelty reports
│   └── validation_embedding/    # BiRNA embedding (optional validation path)
├── scripts/                     # Runnable CLIs by domain → scripts/README.md
├── notebooks/                   # Marimo apps + exported figures
├── tests/                       # pytest (16 tests; no Vienna/NUPACK required)
├── cluster/                     # RunPod, EC2, Macleod EVA runbooks
├── data/                        # Raw reference + processed artefacts (.gitignore policy)
├── typings/                     # Stubs for optional NUPACK / ViennaRNA (ty)
├── .github/workflows/ci.yml     # ruff + ty + pytest
├── Makefile                     # make ci
├── pyproject.toml               # uv lockfile, dependency groups
└── environment.yml              # Conda bio-toolchain (cd-hit, infernal, hmmer)
```

**Split principle:** `src/` holds reusable logic; `scripts/` holds thin orchestration; `notebooks/` are interactive viewers and audit apps (prefer loading JSON sidecars over retraining inline). Paths resolve via `src/data_engineering/paths.py` → `resolve_path()`.

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

## Non-circular feature matrix ($p = 92$)

The legacy 20-column intensive + dynamic set included **$T_m$, Hill, amplitude, $Z$, $\Delta P_{\mathrm{RBS}}$, $\Delta\Delta G$** in $X$, which circularly encoded the melting phenotype we later gate on. Those scalars remain on the fused table for **post-hoc scoring only**.

**In $X$ (92 columns):**

- **7** static 37 °C biophysics (Vienna/NUPACK MFE per nt, ensemble diversity, mean positional entropy, max stem/loop length)
- **3** composition (%GC, length, $P_{\mathrm{paired,RBS}}(37\,^{\circ}\mathrm{C})$)
- **16 + 64** dinucleotide and trinucleotide frequencies (intensive, from matched FASTA)
- **2** SD–AUG spacing + missing-AUG sentinel (`sd_aug_spacing = -1`, `sd_aug_missing`)

Missing AUG is **not** dropped (RefSeq UTRs lack initiators more often than Rfam positives). Feature log: `data/processed/rf_noncircular_feature_log.json`.

---

## Key results

| Check | Result | Sidecar |
|-------|--------|---------|
| Length-alone AUC (legacy) | ≈ 0.20 | `rf_noncircular_diagnostics.json` |
| Stratified CV (20-col circular) | ≈ 0.80 (family leakage) | same |
| **GroupKFold (non-circular $X$)** | **≈ 0.28** | same |
| EVA accepted / yield (2,000 panel) | 105 / 2,000 = **5.25%** | `eva_denovo_checklist.json` |
| Vienna vs NUPACK $T_m$ parity | Gates locked to **ViennaRNA only** | Supplementary Figure 5 narrative |

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
| 6 | [`06_classifier_architecture_ladder.py`](notebooks/06_classifier_architecture_ladder.py) | Circular → non-circular architecture ladder |
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

- **NUPACK 4.1:** manual wheel from `nupack-4.1.0.1/package/` (licensed; gitignored).
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

# Non-circular RF + post-hoc JSON sidecars
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
| `data/processed/fused_features_refseq_dynamic.csv` | Labelled panel (local; not committed) |
| `data/processed/rf_noncircular_diagnostics.json` | GroupKFold AUC + grouped permutation |
| `data/processed/rf_posthoc_report.json` | ŷ bins, gates, Spearman, checklist |
| `data/processed/eva_denovo_checklist.json` | Four-gate pass rates by cohort |
| `notebooks/figures/thesis_figures/` | Chapter 4 export PNGs |
| `notebooks/figures/08_eva_denovo/` | App 08 audit figures |

---

## Thesis markdown package

Thesis prose and supplementary source **do not** live in this repository.

| Location | Contents |
|----------|----------|
| [`../thesis_md_package/`](../thesis_md_package/) | Guideline evals, accuracy audits, supplementary booklet |
| `../thesis_md_package/supplementary/Supplementary_Material.md` | Booklet source (Sections 1–6) |
| `../thesis_md_package/guideline_eval/` | UK word-count, v2–v7 evals, citation syntax |

Rebuild supplementary figures from the thesis package:

```bash
cd ../thesis_md_package/supplementary
PYTHONPATH=../Thermoswitches-MLBIO/src python build_supplementary_figures.py
```

---

## Licence and citation

Thesis: Amier Mohd Zuhri, University of Aberdeen, 2026. Cite Rfam (Kalvari *et al.*, 2021), RefSeq (O'Leary *et al.*, 2016), ViennaRNA (Lorenz *et al.*, 2011), and EVA (GENTEL-Lab checkpoint) as described in the supplementary booklet.

See [`CHANGELOG.md`](CHANGELOG.md) for release notes.
