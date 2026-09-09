[![CI](https://github.com/arcblanc/Thermoswitches-MLBIO/actions/workflows/ci.yml/badge.svg)](https://github.com/arcblanc/Thermoswitches-MLBIO/actions/workflows/ci.yml)

## <a name="started"></a>Getting Started

```sh
git clone https://github.com/arcblanc/Thermoswitches-MLBIO.git
cd Thermoswitches-MLBIO
uv sync --group dev --group notebooks
cp .env.example .env                         # EMAIL, NCBI_API_KEY (optional until downloads)
make test                                    # 16 pytest cases (no Vienna/NUPACK required)

# inspect committed JSON sidecars (no retrain)
python - <<'PY'
import json
d = json.load(open("data/processed/rf_noncircular_diagnostics.json"))
print(d["noncircular_stratified_group"]["roc_auc"])
PY

# open thesis Marimo apps (load sidecars, do not retrain inline)
uv run marimo edit notebooks/01_curated_matched_rfam_refseq_cohort.py
uv run marimo edit notebooks/04_groupkfold_auc_showed_no_transfer.py
uv run marimo edit notebooks/05_eva_yield_gates_then_melting_audit.py

# optional: biophysics engines + full quality gate
uv sync --group biophysics                   # ViennaRNA → import RNA
make ci                                      # ruff + format + uv check + pytest
```

**Note:** NUPACK 4.1 is licensed separately. Install the wheel from `nupack-4.1.0.1/package/` if you need dual-engine sweeps. Yield gates in the thesis path lock to **ViennaRNA only**.

For end-to-end rebuild commands see [`docs/cookbook.md`](docs/cookbook.md). For HPC EVA generation see [`cluster/README.md`](cluster/README.md).

## Table of Contents

- [Getting Started](#started)
- [Users' Guide](#uguide)
  - [Installation](#install)
  - [General usage](#general)
  - [Use cases](#cases)
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

Thermoswitches-MLBIO is an executable analysis stack for *in silico* design of synthetic RNA thermoswitches (RNATs). Typical use cases include:

1. rebuilding or auditing a length/%GC-matched Rfam vs RefSeq 5′ UTR panel ($N = 2{,}396$)
2. training and diagnosing a 92-column Random Forest that withholds melting scalars from $X$
3. running frozen **EVA 1.4B** generation on Macleod and streaming three-gate yield triage
4. biophysical characterisation of the **105** yield-gated candidates against Rfam/RefSeq controls
5. BLAST/nhmmer novelty reporting against Rfam
6. exporting analytical Results figures from committed JSON sidecars

On StratifiedGroupKFold grouped by Rfam accession, the 92-column RF returns AUC ≈ **0.28**, which is below length-alone baselines in some settings and shows that static 37 °C snapshot features do not learn transferable switching rules. After that finding the pipeline pivots to EVA generation (2,000 sequences) and yield triage recovering **105 / 2,000 = 5.25%** under

$$Z \le -2 \land \Delta P_{\mathrm{RBS}} > 0 \land E_{\mathrm{Rfam}} > 10^{-3}.$$

Detailed methods live in [`notebooks/rf_noncircular_methodology.md`](notebooks/rf_noncircular_methodology.md). Melting-audit findings live in [`notebooks/05_eva_melting_summarised_findings.md`](notebooks/05_eva_melting_summarised_findings.md).

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

Paths resolve through `src/data_engineering/paths.py` → `resolve_path()`. Do not commit `.venv`.

### <a name="general"></a>General usage

Library code lives under `src/`. Thin CLIs live under `scripts/` (see [`scripts/README.md`](scripts/README.md)). Interactive audit apps are Marimo scripts under `notebooks/` (see [`notebooks/README.md`](notebooks/README.md)). They **load JSON/CSV sidecars** and do not retrain models inline.

```sh
# CLI pattern used in playbooks
PYTHONPATH=src python scripts/<category>/<script>.py [args]

# quality gate matching GitHub Actions
make ci
```

Committed lightweight sidecars under `data/processed/*.json` are the audit source of truth. Large CSVs, FASTA pools, and `.joblib` models are gitignored and must be regenerated locally (see [Data artefacts](#data)).

### <a name="cases"></a>Use cases

#### <a name="cohort"></a>Audit the matched Rfam / RefSeq cohort

```sh
uv run marimo edit notebooks/01_curated_matched_rfam_refseq_cohort.py
uv run marimo edit notebooks/02_matching_removed_compositional_shortcuts.py
```

App 01 loads the fused panel and checks length/%GC matching, 37 °C biophysics, SD–AUG geometry, and k-mer dialect. App 02 shows composition-only AUCs $< 0.50$ after Hungarian matching — the shortcut kill required before RF claims.

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

App 06 does **not** re-run BLAST/nhmmer. It loads `data/processed/novelty/*.csv` and `*.json` and buckets identical / remote-homolog / no-hit outcomes.

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

Open a GitHub issue for reproducible bugs (include command, sidecar path, and `uv` / OS versions). There is not a separate mailing list.

### <a name="cite"></a>Citing this work

Amier Mohd Zuhri (2026). *In Silico De Novo Design and Validation of Synthetic RNA Thermoswitches Via Deep Generative Modelling*. MSc Data Science, University of Aberdeen. Repository: <https://github.com/arcblanc/Thermoswitches-MLBIO>.

If you use ViennaRNA, NUPACK, EVA, Rfam, or Infernal/HMMER, cite those tools separately.

## <a name="dguide"></a>Developers' Guide

Editable install and CI:

```sh
uv sync --group dev --group biophysics --group notebooks --group cloud --group llm
make lint && make format-fix && make typecheck && make test
```

| Package | Responsibility |
|---------|----------------|
| `data_engineering` | Rfam extract, RefSeq UTRs, CD-HIT, length/%GC match |
| `thermo_sim` | Vienna/NUPACK, RF, post-hoc, batch, thesis figures |
| `de_novo_hallucinations` | EVA (+ GenerRNA appendix) wrappers, quality gates |
| `novelty_eval` | BLAST / nhmmer parsing |
| `validation_embedding` | Optional BiRNA embedding path |

Type stubs for optional engines live in `typings/`. Prefer adding logic under `src/` and thin wrappers under `scripts/<category>/`. Marimo apps should load sidecars rather than embed training loops.

## <a name="limit"></a>Limitations

- The 92-column RF is a ranking diagnostic, not a wet-lab gate. GroupKFold AUC ≈ 0.28 means it does **not** transfer thermoswitch rules across Rfam families.
- Yield triage admits structured RNA under $Z$, $\Delta P_{\mathrm{RBS}}$, and Rfam $E$-value. It does not guarantee Neupert-spec heat-shock controllers (Tier 2 survival is rare even among natural Rfam).
- Dual-engine $T_m$ can disagree, so thesis gates lock to ViennaRNA.
- EVA weights and large FASTA/CSV artefacts are not shipped in git. GenerRNA source is retained as an appendix path. Local weights are optional and large.
- Marimo apps assume committed sidecars under `data/processed/`. Missing CSVs will fail cells that regenerate from fused panels.
