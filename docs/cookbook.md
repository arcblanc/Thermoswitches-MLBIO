# Cookbook

End-to-end recipes for Thermoswitches-MLBIO. Short conceptual usage stays in the [root README](../README.md). This file is for full rebuilds and expected artefact paths.

**Prerequisites:** `uv sync` with the groups you need. Run every command from the **repo root**.

## 1. Smoke the install

```sh
uv sync --group dev --group notebooks
make test
python - <<'PY'
import json
d = json.load(open("data/processed/rf_noncircular_diagnostics.json"))
print(d["noncircular_stratified_group"]["roc_auc"])
PY
```

Expected: 16 passed tests and a printed GroupKFold AUC ≈ 0.28.

## 2. Rebuild the matched panel (expensive)

Positives (Rfam):

```sh
python src/data_engineering/data_extraction.py
python src/data_engineering/sequence_retrieval.py --all
python src/data_engineering/cd_hit_sequence_similarity.py --all
```

Negatives (RefSeq 5′ UTRs) + match:

```sh
bash scripts/extraction/download_refseq_genomes.sh
PYTHONPATH=src python src/data_engineering/refseq_utr_extract.py
PYTHONPATH=src python src/data_engineering/cmscan_decontaminate.py
PYTHONPATH=src python src/data_engineering/length_gc_match.py \
  --all-positives --cds-truncate \
  --negatives-csv data/processed/refseq_utr/candidates_clean.csv \
  --negatives-fasta data/processed/refseq_utr/candidates_clean.fasta
```

Thermo batch + enrich → fused table:

```sh
uv sync --group biophysics
python src/thermo_sim/thermo_batch.py --run
PYTHONPATH=src python src/thermo_sim/enrich_dynamic_features.py --workers 4
```

**Output:** `data/processed/fused_features_refseq_dynamic.csv` (gitignored, $N = 2396$).

## 3. Retrain the noncircular RF and write sidecars

```sh
python src/thermo_sim/thermo_classifier.py train \
  --fused-csv data/processed/fused_features_refseq_dynamic.csv \
  --model-path data/processed/models/rf_thermoswitch_noncircular.joblib

python src/thermo_sim/thermo_classifier.py posthoc \
  --fused-csv data/processed/fused_features_refseq_dynamic.csv

PYTHONPATH=src python scripts/rf/rf_length_bias_diagnostics.py --noncircular
```

**Outputs (committed when you choose to update them):**

- `data/processed/rf_noncircular_diagnostics.json`
- `data/processed/rf_posthoc_report.json`
- `data/processed/rf_noncircular_feature_log.json`
- `data/processed/models/rf_thermoswitch_noncircular.joblib` (gitignored)

Groups for honesty CV: Rfam accession (positives), `REFSEQ:{assembly}` (negatives).

## 4. EVA generation on Macleod (cluster)

Follow [`cluster/MACLEOD_EVA.md`](../cluster/MACLEOD_EVA.md) and [`cluster/EVA_OVERVIEW.md`](../cluster/EVA_OVERVIEW.md). High-level flow:

1. Install / smoke EVA on `gpu02` (`scripts/eva/`).
2. Generate TaxID-conditioned `mRNA` prompts → `generated.fasta` (~2,000 sequences).
3. Pull FASTA to the analysis machine.

Local streaming triage:

```sh
PYTHONPATH=src python scripts/triage/eva_stream_triage.py --source ssh
PYTHONPATH=src python scripts/triage/eva_yield_ratio.py
```

**Locked yield rule:** $Z \le -2 \land \Delta P_{\mathrm{RBS}} > 0 \land E_{\mathrm{Rfam}} > 10^{-3}$ (ViennaRNA). Thesis yield: **105 / 2,000 = 5.25%**. Checklist sidecar: `data/processed/eva_denovo_checklist.json`.

## 5. Melting audit of the 105 passers

```sh
uv run marimo edit notebooks/05_eva_yield_gates_then_melting_audit.py
```

Library entry points: `src/thermo_sim/eva_denovo_characterization.py`. Figures land in `notebooks/figures/05_eva_melting/`. Written brief: [`notebooks/05_eva_melting_summarised_findings.md`](../notebooks/05_eva_melting_summarised_findings.md).

## 6. Novelty search vs Rfam

```sh
bash scripts/extraction/download_rfam.sh
bash scripts/extraction/run_novelty_search.sh
uv run marimo edit notebooks/06_candidates_vs_rfam_novelty.py
```

**Outputs:** `data/processed/novelty/novelty_report.csv`, `novelty_summary.json`, and related TSV/OUT files.

## 7. Export analytical Chapter 4 figures

```sh
uv run marimo edit notebooks/07_export_analytical_thesis_figures.py
```

**Outputs:** `notebooks/figures/thesis_figures/04_*.png` and related analytical panels (no schematic flowcharts).

## 8. Composition-only length/%GC Random Forests (proxy memorisation check)

```sh
uv run marimo edit notebooks/02_matching_removed_compositional_shortcuts.py
```

Reads `data/processed/rf_compositional_ablation.json`.
Composition-only AUCs $< 0.50$ on the matched panel means length and %GC alone no longer separate the classes.

## Interpretation checklist

| Step | Success signal |
|------|----------------|
| Install | `make test` green |
| Matching | App 02 composition AUCs $< 0.50$ |
| Noncircular RF | GroupKFold AUC ≈ 0.28 in diagnostics JSON |
| EVA triage | ~5% yield (105 passers for the thesis panel) |
| Melting | Flat 37–42 °C RBS exposure relative to FourU-class controls |
| Novelty | Category counts in `novelty_summary.json` |

If a step fails, see [`FAQ.md`](../FAQ.md) before changing gate thresholds.
