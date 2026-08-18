# Non-circular Random Forest — model update and results

**Date:** 18 August 2026  
**Panel:** `fused_features_refseq_dynamic.csv` (n = 2,396; 1,198 Rfam positives / 1,198 RefSeq-matched 5′ UTR negatives)  
**Model:** `data/processed/models/rf_thermoswitch_noncircular.joblib` (sklearn RF, 200 trees, trained 2026-08-18)  
**Architecture notebook:** [`07_classifier_architecture_ladder.ipynb`](07_classifier_architecture_ladder.ipynb) (§§1–3 history; §§4–7 this model)  
**Executable tables:** [`08_noncircular_rf_results.ipynb`](08_noncircular_rf_results.ipynb)

This note replaces the circular 20-column RF as the production ranking model. It does **not** replace the locked EVA yield gate.

---

## 1. What changed

The previous intensive + dynamic RF put **Tm, Hill coefficient, amplitude, MFE Z-score, ΔP_RBS, and ΔΔG** in \(X\). Those scalars *are* the melting phenotype the ranker is supposed to help find, so the forest was scoring the answer.

The new forest trains only on **non-circular** inputs measured at 37 °C plus sequence composition. Melting scalars stay on the fused table and are applied **after** \(\hat{y}\).

| Role | Contents |
|------|----------|
| **\(X\) (92 columns)** | Static 37 °C physics (Vienna/NUPACK `MFE_per_nt`, ensemble diversity, positional entropy, max stem/loop); `%GC`; length; \(P_{\mathrm{paired,RBS}}(37^\circ\mathrm{C})\); 16 dinucleotide + 64 trinucleotide **frequencies**; SD-to-AUG spacer |
| **Excluded from \(X\)** | `Tm`, `hill_coeff`, `amplitude`, `viennarna_mfe_zscore`, `viennarna_delta_P_RBS`, `viennarna_delta_delta_G` |
| **Post-hoc** | Confidence bins, ΔP_RBS, \(n_H\), \(T_m\), \(Z\), Vienna–NUPACK Spearman, MW/KS, visual checklist |

`--circular-features` still trains the old 20-column RF for comparison. XGBoost (`train-xgb`) is unchanged on that circular set.

---

## 2. Design choices that affect the numbers

**Missing AUG is a sentinel, not a dropped row.** RefSeq negatives lack an initiator more often than Rfam positives (590 vs 475). Dropping those rows would have shrunk the negative class. Encoding: `sd_aug_spacing = -1` and `sd_aug_missing = 1`. Fit size stayed **2,396 / 1,198 / 1,198**.

**k-mers are frequencies**, not raw counts, so length does not return as a count proxy.

**Attribution is grouped permutation**, not Gini/MDI. Sixty-four trinucleotides would dilute impurity across correlated motifs. Groups: static biophysics, composition, dinucleotides, trinucleotides, SD–AUG. The permutation numbers below are from a **full-fit** forest (in-sample AUC = 1.0) — they say what the overfit model uses, not what transfers.

**High-bin Spearman requires \(N \ge 25\).** GroupKFold produced only 6 scores \(\ge 0.80\), so high-bin \(r_s\) is logged as underpowered. Panel-wide Spearman is the primary cross-engine metric.

---

## 3. CV results

Same honesty split as notebook 07 §2: StratifiedKFold (family leakage) vs StratifiedGroupKFold on `rfam_acc` / `REFSEQ:{assembly}`.

| Model | Random-split AUC | Family-holdout AUC | Read as |
|-------|------------------|--------------------|---------|
| Length alone (control) | 0.20 | — | Length shortcut is gone |
| Circular 20-col RF (history) | ~0.80 | **0.19** | No out-of-family detector |
| **Non-circular RF (this model)** | **0.97** | **0.28** | Motifs leak family on a random split; holdout is still near chance |

Sources: `data/processed/rf_noncircular_diagnostics.json`; circular history in `refseq_dynamic_rf_diagnostics.json` / notebook 07.

**Takeaway.** Removing circular melting scalars and adding k-mers did **not** produce a transferable thermoswitch detector. Random-split 0.97 is family-shaped k-mer memory. Holdout 0.28 is a small lift over 0.19, not a usable classifier. The RF remains a **ranking aid**, not a wet-lab gate.

---

## 4. Grouped permutation importance (in-sample)

| Block | n features | Mean AUC drop |
|-------|------------|---------------|
| Trinucleotides | 64 | 0.338 |
| Dinucleotides | 16 | 0.004 |
| Static 37 °C physics | 7 | ~0 |
| Composition (GC, length, \(P_{\mathrm{paired,37}}\)) | 3 | 0 |
| SD–AUG (spacing + missing flag) | 2 | 0 |

The overfit forest lives in the 64-mer block. That is consistent with Stratified AUC 0.97 and GroupKFold 0.28: local sequence motifs separate Rfam families when the family is seen at train time, and do not carry a general switch physics.

Figure: [`figures/07_classifier/grouped_permutation_importance.png`](figures/07_classifier/grouped_permutation_importance.png) (regenerated from notebook 07 §5).

---

## 5. Post-hoc gates (out-of-fold \(\hat{y}\))

Bins from StratifiedGroupKFold probabilities (`rf_posthoc_report.json`):

| Bin | Rule | n |
|-----|------|---|
| High | \(\hat{y} \ge 0.80\) | **6** |
| Mid | \(0.40 < \hat{y} < 0.60\) | 297 |
| Low | \(\hat{y} \le 0.20\) | 1,504 |
| Other | remainder | 589 |

Almost nobody is high-confidence out of family — another view of the 0.28 holdout.

Panel-wide melting filters (count of rows passing; not stacked):

| Gate | n pass / 2,396 |
|------|----------------|
| \(\Delta P_{\mathrm{RBS}} > 0\) | 2,294 |
| Vienna \(n_H > 1.0\) | 2,377 |
| \(T_m \in [42, 45]^\circ\mathrm{C}\) | 168 |
| \(Z \le -2\) | 1,128 |

**Cross-engine consensus (primary = panel-wide Spearman, n = 2,396):**

| Pair | \(r_s\) | p |
|------|---------|---|
| Vienna vs NUPACK \(T_m\) | 0.035 | 0.083 |
| Hill | 0.037 | 0.072 |
| Amplitude | 0.033 | 0.103 |

The two engines barely rank-agree on this panel. High-bin \(r_s\) was **not reported** (\(N = 6 < 25\)).

**High vs low bins (treat as exploratory: \(n_{\mathrm{high}} = 6\)):** Mann–Whitney / KS on \(T_m\) reach \(p \approx 0.007\) / 0.002; Hill, amplitude, \(Z\), and ΔP_RBS do not. With six high scores this is not a stable phenotype contrast.

---

## 6. Visual diagnostic checklist

Four signatures for a melting curve (notebook 07 §7; [`figures/07_classifier/melting_visual_checklist.png`](figures/07_classifier/melting_visual_checklist.png)):

| Signature | Pass rule | n pass on labelled panel |
|-----------|-----------|--------------------------|
| Sigmoidal snap | \(n_H > 1.5\) | 2,355 |
| Inflection | \(T_m\) 42–45 °C | 168 |
| Dynamic range | \(\Delta\theta \ge 0.50\) | 699 |
| Baseline repression | \(P_{\mathrm{open,RBS}}(37^\circ\mathrm{C}) \le 0.20\) | 75 |
| **All four** | intersection | **0** |

Hill fits on this panel are steep almost everywhere; the heat-shock \(T_m\) window and a locked 37 °C RBS are rare, and they do not co-occur with a large amplitude. Post-hoc numeric cooperativity is \(n_H > 1.0\); \(n_H > 1.5\) is the visual “snap.”

---

## 7. What this model is for

- **Use** the non-circular RF to rank sequences when you want a score that is not tautological with the melting curve.
- **Do not** treat \(\hat{y} \ge 0.5\) (or 0.80) as a thermoswitch call. Out-of-family AUC is 0.28; the high bin has six sequences.
- **Yield of EVA RNA is unchanged:** \(Z \le -2 \land \Delta P_{\mathrm{RBS}} > 0 \land E_{\mathrm{Rfam}} > 10^{-3}\). Stream FASTA still has no Hill/\(T_m\); stacking the full post-hoc checklist on generated RNA needs `thermo_batch`.

---

## 8. Reproduce

```bash
PYTHONPATH=src python src/thermo_sim/enrich_dynamic_features.py --p-open-only --workers 4

python src/thermo_sim/thermo_classifier.py train \
  --fused-csv data/processed/fused_features_refseq_dynamic.csv \
  --model-path data/processed/models/rf_thermoswitch_noncircular.joblib

python src/thermo_sim/thermo_classifier.py posthoc \
  --fused-csv data/processed/fused_features_refseq_dynamic.csv
```

| Artifact | Contents |
|----------|----------|
| `data/processed/models/rf_thermoswitch_noncircular.joblib` | Fitted forest + `feature_set: noncircular` |
| `data/processed/models/rf_thermoswitch_noncircular.json` | Sidecar: 92 columns, n = 2,396, circular exclusions |
| `data/processed/rf_noncircular_feature_log.json` | X list, AUG-missing rates by class |
| `data/processed/rf_noncircular_diagnostics.json` | AUCs + grouped permutation |
| `data/processed/rf_posthoc_report.json` | Bins, gates, Spearman, MW/KS, checklist |
| `logs/rf_noncircular.log` | Train + posthoc stdout |
