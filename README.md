[![CI](https://github.com/arcblanc/Thermoswitches-MLBIO/actions/workflows/ci.yml/badge.svg)](https://github.com/arcblanc/Thermoswitches-MLBIO/actions/workflows/ci.yml)

## <a name="started"></a>Getting Started

```sh
git clone https://github.com/arcblanc/Thermoswitches-MLBIO.git
cd Thermoswitches-MLBIO
uv sync --group dev --group notebooks
cp .env.example .env                         # EMAIL, NCBI_API_KEY (optional until downloads)
make test                                    # 16 pytest cases (no Vienna/NUPACK required)

# Loads finished runs - for eg. a groupkfold AUC -> rf_noncircular_diagnostics.json
python - <<'PY'
import json
d = json.load(open("data/processed/rf_noncircular_diagnostics.json"))
print(d["noncircular_stratified_group"]["roc_auc"])
PY

# open thesis apps (same stage order as below)
uv run marimo edit notebooks/01_curated_matched_rfam_refseq_cohort.py
uv run marimo edit notebooks/04_groupkfold_auc_showed_no_transfer.py
uv run marimo edit notebooks/05_eva_yield_gates_then_melting_audit.py

# optional: biophysics engines + full quality gate
uv sync --group biophysics                   # ViennaRNA → import RNA
make ci                                      # project gate: core ruff + ty + pytest
make cleanliness                             # local: also lint/format notebooks
```

**Note:** This project can be enacted with dual engine support. NUPACK 4.1 is licensed separately. Install the wheel from `nupack-4.1.0.1/package/` if you need dual-engine sweeps. However, in the thesis, the yield gates in the thesis path lock to **ViennaRNA only**.

For end-to-end rebuild commands see [`docs/cookbook.md`](docs/cookbook.md). For HPC EVA generation see [`cluster/README.md`](cluster/README.md).

## Table of Contents

- [Getting Started](#started)
- [Users' Guide](#uguide)
  - [Installation](#install)
  - [General usage](#general)
  - [Pipeline stages](#stages)
    - [Audit the matched Rfam / RefSeq cohort](#cohort)
    - [Inspect the honest Random Forest](#rf)
    - [Triage EVA FASTA and audit melting](#eva)
    - [Check candidate novelty vs Rfam](#novelty)
    - [Export analytical thesis figures](#figures)
  - [Pipeline overview](#pipeline)
  - [Feature matrix (p = 92)](#features)
  - [Data artefacts](#data)
  - [Getting help](#help)
  - [Citing this work](#cite)
- [Developers' Guide](#dguide)
- [Limitations](#limit)

## <a name="uguide"></a>Users' Guide

Thermoswitches-MLBIO is the code I used for *in silico* design of synthetic RNA thermoswitches (RNATs). **Marimo** is a reactive Python notebook tool: each file under `notebooks/` is an app that opens in the browser, loads committed JSON/CSV sidecars, and walks one stage of the thesis path without retraining models inline. The layout follows that same order.

```text
length/%GC-matched Rfam vs RefSeq 5′ UTR panel (N = 2,396)
        │
        ▼
composition-only Random Forests (length and %GC alone)
test for length/%GC proxy memorisation → AUC < 0.50
        │
        ▼
92-column Random Forest
melting scalars kept out of feature matrix X
StratifiedGroupKFold by Rfam accession
        │
        ▼
GroupKFold AUC ≈ 0.28
(static 37 °C features did not transfer across Rfam families)
        │
        ▼
frozen EVA 1.4B on Macleod → 2,000 sequences
        │
        ▼
yield filters (all three required):
  Z ≤ −2
  ΔP_RBS > 0
  E_Rfam > 10⁻³
        │
        ▼
105 sequences pass
        │
        ├── melting curves vs Rfam / RefSeq controls
        ├── BLAST / nhmmer novelty vs Rfam
        └── Results figures from committed JSON sidecars
```

What I did, in thesis order:

1. I built a length/%GC-matched panel of Rfam thermoswitch 5′ UTRs against RefSeq controls ($N = 2{,}396$).
   After matching I checked folding energy, SD–AUG geometry, and k-mer composition between the two classes.
2. I trained Random Forests that used **only** length and %GC on that matched panel.
   The reason was to test whether the classifier could still separate classes by memorising length and %GC as proxies for class, which matching was meant to remove.
   Those composition-only models returned AUCs $< 0.50$.
3. I trained a 92-column Random Forest on static 37 °C biophysics, dinucleotide and trinucleotide frequencies, and SD–AUG spacing.
   Melting scalars ($T_m$, Hill coefficient, $Z$, $\Delta P_{\mathrm{RBS}}$) stayed out of the feature matrix $X$ because those same quantities are applied later as yield filters.
4. Under StratifiedGroupKFold grouped by Rfam accession, that model returned AUC ≈ **0.28**.
   Family-held-out transfer failed, so I moved on from the Random Forest as a portable switch detector.
5. I generated 2,000 sequences with frozen **EVA 1.4B** on Macleod.
   Sequences had to pass all three yield filters
   $$
   Z \le -2 \land \Delta P_{\mathrm{RBS}} > 0 \land E_{\mathrm{Rfam}} > 10^{-3},
   $$
   which left **105 / 2,000 = 5.25%**.
6. I compared melting behaviour of those 105 sequences to Rfam and RefSeq controls (Hill coefficient, $T_m$, $P_{\mathrm{open}}$).
   I scored novelty against Rfam with BLAST and nhmmer.
   I exported Results figures from the committed JSON sidecars.

Methods detail: [`notebooks/rf_noncircular_methodology.md`](notebooks/rf_noncircular_methodology.md). Melting-audit findings: [`notebooks/05_eva_melting_summarised_findings.md`](notebooks/05_eva_melting_summarised_findings.md).

### <a name="install"></a>Installation

You need [uv](https://docs.astral.sh/uv/) and Python 3.12+. From the repo root:

```sh
uv sync --group dev --group notebooks
```

Common optional groups:

| Group | Provides |
|-------|----------|
| `biophysics` | ViennaRNA (`import RNA`) |
| `cloud` | boto3 / SSH helpers for EC2–S3 batches |
| `llm` | torch / transformers for EVA or GenerRNA paths |
| `dev` | ruff, pytest, ty |

For cd-hit, Infernal, and HMMER on a workstation or cluster head node:

```sh
conda env create -f environment.yml
```

Paths resolve through `src/data_engineering/paths.py` → `resolve_path()`. 

### <a name="general"></a>General usage

Reusable library code lives under `src/`. Command-line entry points under `scripts/` are short wrappers that call into `src/` (argument parsing, paths, and orchestration only — see [`scripts/README.md`](scripts/README.md)). Thesis stages live under `notebooks/` (see [`notebooks/README.md`](notebooks/README.md)).

```sh
# CLI pattern used in playbooks
PYTHONPATH=src python scripts/<category>/<script>.py [args]

# quality gate matching GitHub Actions
make ci
```

Committed lightweight sidecars under `data/processed/*.json` are the audit source of truth. Large CSVs, FASTA pools, and `.joblib` models are gitignored and must be regenerated locally (see [Data artefacts](#data)).

### <a name="stages"></a>Pipeline stages

Each subsection is one stretch of the same analysis path (Marimo apps 01–07).

#### <a name="cohort"></a>Audit the matched Rfam / RefSeq cohort

```sh
uv run marimo edit notebooks/01_curated_matched_rfam_refseq_cohort.py
uv run marimo edit notebooks/02_matching_removed_compositional_shortcuts.py
```

App 01 loads the fused panel.
It checks length/%GC matching, 37 °C biophysics, SD–AUG geometry, and k-mer dialect on the matched cohort.

App 02 trains Random Forests that use **only** length and %GC after Hungarian matching.
I ran that check to see whether the model could still separate Rfam from RefSeq by memorising length and %GC as class proxies.
Matching was built to remove that length/%GC route.
Those composition-only models returned AUCs $< 0.50$.

#### <a name="rf"></a>Inspect the honest Random Forest

```sh
uv run marimo edit notebooks/03_rf_honesty_noncircular_matrix.py
uv run marimo edit notebooks/04_groupkfold_auc_showed_no_transfer.py
PYTHONPATH=src python scripts/rf/rf_length_bias_diagnostics.py --noncircular
```

Melting scalars ($T_m$, Hill, $Z$, $\Delta P_{\mathrm{RBS}}$) stay out of $X$ and enter post-hoc ranking only. StratifiedGroupKFold on `rfam_acc` / `REFSEQ:{assembly}` is the honesty protocol. Expect GroupKFold AUC ≈ 0.28 in `data/processed/rf_noncircular_diagnostics.json`.

#### <a name="eva"></a>Triage EVA FASTA and audit melting

```sh
# streaming yield triage (example: SSH source from Macleod)
PYTHONPATH=src python scripts/triage/eva_stream_triage.py --source ssh

uv run marimo edit notebooks/05_eva_yield_gates_then_melting_audit.py
```

Generation itself runs on Macleod (`cluster/MACLEOD_EVA.md`). Local machines consume FASTA + triage tables, then App 05 compares Hill / $T_m$ / $P_{\mathrm{open}}$ of the 105 passers to Rfam and RefSeq. It is usually recommended to keep gates on ViennaRNA when NUPACK and Vienna disagree on $T_m$.

#### <a name="novelty"></a>Check candidate novelty vs Rfam

```sh
bash scripts/extraction/download_rfam.sh          # if regenerating
bash scripts/extraction/run_novelty_search.sh
uv run marimo edit notebooks/06_candidates_vs_rfam_novelty.py
```

App 06 loads `data/processed/novelty/*.csv` and `*.json`.
It buckets identical / remote-homolog / no-hit outcomes from the committed BLAST and nhmmer reports.

#### <a name="figures"></a>Export analytical thesis figures

```sh
uv run marimo edit notebooks/07_export_analytical_thesis_figures.py
```

Writes ROC, permutation, attrition funnel, and Hill overlay PNGs under `notebooks/figures/thesis_figures/`. Flowchart schematics are intentionally omitted.

### <a name="pipeline"></a>Pipeline overview

```
Rfam positives (1,198) ──┐
                         ├── CD-HIT 80% ── length/GC match ── Vienna + NUPACK ── 92-col X ── RF (200 trees)
RefSeq 5′ UTRs (1,198) ──┘                              │              │
                                                        │              └── OOF ŷ bins (post-hoc only)
                                                        └── Tm, Hill, Z, ΔP_RBS (withheld from X)

EVA 1.4B (frozen, TaxID-conditioned mRNA) ── Macleod gpu02 ── generated.fasta (2,000)
        │
Mac CPU ── Vienna Z / ΔP_RBS + Rfam novelty ── yield gate ── 105 candidates ── App 05 melting audit
```

| Stage | Role | Primary code |
|-------|------|----------------|
| Corpus | Rfam vs matched RefSeq 5′ UTRs | `src/data_engineering/` |
| Features ($X$) | Static 37 °C physics + k-mers + SD–AUG | `src/thermo_sim/noncircular_features.py` |
| RF | Bootstrap ensemble + grouped CV | `src/thermo_sim/thermo_classifier.py` |
| Post-hoc | ŷ bins, gates, Spearman, checklist | `src/thermo_sim/rf_posthoc.py` |
| EVA | Frozen CLM generation | `src/de_novo_hallucinations/eva_generate.py` |
| Triage | Three-gate yield on streaming FASTA | `scripts/triage/` |
| Audit | EVA vs Rfam/RefSeq melting | `src/thermo_sim/eva_denovo_characterization.py` |

### <a name="features"></a>Feature matrix ($p = 92$)

**In $X$:** 7 static 37 °C biophysics + 3 composition + 16 dinucleotide + 64 trinucleotide frequencies + 2 SD–AUG spacing/sentinel columns. **Out of $X$:** $T_m$, Hill coefficient, $Z$, $\Delta P_{\mathrm{RBS}}$ (post-hoc only). Missing AUG is retained with a sentinel (`sd_aug_spacing = -1`), because dropping those rows would unbalance RefSeq negatives. Feature log: `data/processed/rf_noncircular_feature_log.json`.

### <a name="data"></a>Data artefacts

Large regenerated tables are gitignored. Lightweight JSON sidecars are committed for audit:

| Path | Contents |
|------|----------|
| `data/processed/rf_noncircular_diagnostics.json` | GroupKFold AUC + permutation |
| `data/processed/rf_posthoc_report.json` | ŷ bins, gates, Spearman, checklist |
| `data/processed/eva_denovo_checklist.json` | Four-gate pass rates by cohort |
| `data/processed/novelty/` | BLAST / nhmmer reports |
| `notebooks/figures/thesis_figures/` | Analytical Chapter 4 PNGs |
| `notebooks/figures/05_eva_melting/` | App 05 melting / PoC figures |

Prototype 4-sequence fixture: `data/processed/prototype/`.

### <a name="help"></a>Getting help

- Troubleshooting: [`FAQ.md`](FAQ.md)
- Full rebuild recipes: [`docs/cookbook.md`](docs/cookbook.md)
- CLI catalogue: [`scripts/README.md`](scripts/README.md)
- Marimo app map: [`notebooks/README.md`](notebooks/README.md)
- Macleod / cloud runbooks: [`cluster/README.md`](cluster/README.md)
- Release notes: [`CHANGELOG.md`](CHANGELOG.md)

Open a GitHub issue for reproducible bugs (include command, sidecar path, and `uv` / OS versions).

### <a name="cite"></a>Citing this work

Amier Mohd Zuhri (2026). *In Silico De Novo Design and Validation of Synthetic RNA Thermoswitches Via Deep Generative Modelling*. MSc Data Science, University of Aberdeen. Repository: <https://github.com/arcblanc/Thermoswitches-MLBIO>.

If you use ViennaRNA, NUPACK, EVA, Rfam, or Infernal/HMMER, cite those tools separately.

## <a name="dguide"></a>Developers' Guide

Project gate (matches GitHub Actions):

```sh
uv sync --group dev
make ci
```

Local workstation install (Vienna, notebooks, cloud, LLM) and repo hygiene on Marimo apps:

```sh
make sync-full
make cleanliness
```

| Package | Responsibility |
|---------|----------------|
| `data_engineering` | Rfam extract, RefSeq UTRs, CD-HIT, length/%GC match |
| `thermo_sim` | Vienna/NUPACK, RF, post-hoc, batch, thesis figures |
| `de_novo_hallucinations` | EVA (+ GenerRNA appendix) wrappers, quality gates |
| `novelty_eval` | BLAST / nhmmer parsing |
| `validation_embedding` | Optional BiRNA embedding path |

Type stubs for optional engines live in `typings/`.
Reusable logic belongs under `src/`.
Short command-line wrappers belong under `scripts/<category>/`.
Apps under `notebooks/` load committed sidecars for audit.

## <a name="limit"></a>Limitations

- The 92-column Random Forest is a ranking diagnostic on static 37 °C features.
  StratifiedGroupKFold grouped by Rfam accession returned AUC ≈ **0.28**.
- Yield triage keeps sequences that pass $Z \le -2$, $\Delta P_{\mathrm{RBS}} > 0$, and $E_{\mathrm{Rfam}} > 10^{-3}$.
  Tier 2 Neupert-style heat-shock checklist survival is rare on both the 105 EVA passers and natural Rfam.
- ViennaRNA and NUPACK can return different $T_m$ values on the same sequence.
  Thesis yield filters therefore lock to ViennaRNA.
- Git ships lightweight JSON sidecars under `data/processed/`.
  EVA weights, large FASTA pools, and fused CSVs stay local or on the cluster.
  GenerRNA source remains in-tree as an appendix path.
- Apps under `notebooks/` read committed sidecars under `data/processed/`.
  Cells that rebuild from fused panels need those CSVs present locally.
