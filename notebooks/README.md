# Notebooks

Thesis stages are Marimo `.py` apps under this folder.
Each app loads JSON/CSV sidecars under `data/processed/`.

```sh
uv run marimo edit notebooks/<app>.py
```

## App order (what I did → what I found)

| # | File | What I did | What I found |
|---|------|------------|--------------|
| 01 | [`01_curated_matched_rfam_refseq_cohort.py`](01_curated_matched_rfam_refseq_cohort.py) | Matched Rfam to RefSeq on length/%GC | Cohort still differs in folding / geometry / dialect |
| 02 | [`02_matching_removed_compositional_shortcuts.py`](02_matching_removed_compositional_shortcuts.py) | Trained length/%GC-only Random Forests to test proxy memorisation | Composition AUCs $< 0.50$ |
| 03 | [`03_rf_honesty_noncircular_matrix.py`](03_rf_honesty_noncircular_matrix.py) | Withheld melting scalars from $X$ and used StratifiedGroupKFold | 92-column noncircular protocol locked |
| 04 | [`04_groupkfold_auc_showed_no_transfer.py`](04_groupkfold_auc_showed_no_transfer.py) | Scored noncircular RF sidecars | GroupKFold AUC ≈ **0.28** → move to EVA |
| 05 | [`05_eva_yield_gates_then_melting_audit.py`](05_eva_yield_gates_then_melting_audit.py) | Applied yield filters 2,000 → 105 and ran melting audit | Flat 37–42 °C RBS exposure. Tier 2 rare |
| 06 | [`06_candidates_vs_rfam_novelty.py`](06_candidates_vs_rfam_novelty.py) | Loaded BLAST / nhmmer reports | Homology buckets for the de novo claim |
| 07 | [`07_export_analytical_thesis_figures.py`](07_export_analytical_thesis_figures.py) | Exported analytical Results panels | ROC, permutation, funnel, Hill |

## Figure directories

| Directory | Producer |
|-----------|----------|
| `figures/02_compositional_ablation/` | App 02 |
| `figures/03_rf_honesty/` | Apps 03–04 |
| `figures/05_eva_melting/` | App 05 |
| `figures/thesis_figures/` | App 07 |

## Companion markdown

- [`rf_noncircular_methodology.md`](rf_noncircular_methodology.md) — RF honesty protocol and formulae
- [`05_eva_melting_summarised_findings.md`](05_eva_melting_summarised_findings.md) — melting-audit brief

Helpers for App 01 live in `src/thermo_sim/curation_eda.py`. Figure builders for App 07 live in `src/thermo_sim/thesis_results_figures.py`.
