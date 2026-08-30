# Methodology: Non-Circular Random Forest for RNA Thermoswitch Ranking

**Purpose:** reproducible Methods write-up for the supervised Random Forest enacted in Thermoswitches-MLBIO  
**Primary model:** `data/processed/models/rf_thermoswitch_noncircular.joblib`  
**Panel:** `data/processed/fused_features_refseq_dynamic.csv` ($N = 2396$)  
**Implementation:** `src/thermo_sim/thermo_classifier.py`, `noncircular_features.py`, `rf_posthoc.py`  
**Architecture / results:** notebooks `06`, `07`; update note `07_noncircular_rf_model_update.md`

> Can a Random Forest trained only on **non-circular** 37 °C physics, composition, and $k$-mers rank thermoswitch-like sequences without tautologically encoding the melting phenotype it is meant to help discover?

<br>

**What this document is for**:
<br>
<br>
- Define the labelled panel and matching rules,
<br>
- define the feature matrix $X$ and label $y$,
<br>
- state the ensemble equations and scikit-learn parameters,
<br>
- state the honesty CV protocol and post-hoc gates,
<br>
- give commands a peer can re-run.

<br>

**What this model is not**:
<br>
<br>
- a wet-lab thermoswitch call ($\hat{y} \ge 0.5$ is **not** a gate),
<br>
- a replacement for EVA yield triage ($Z$, $\Delta P_{\mathrm{RBS}}$, $E_{\mathrm{Rfam}}$).

---

## 1. Problem statement and design principle

RNA thermometers (thermoswitches) are structured 5′ UTRs that sequester the Shine–Dalgarno (SD) / RBS at physiological temperature and expose it under heat shock.

<br>

**Circular failure mode (historical 20-column RF)**:
<br>
<br>
- $X$ included $T_m$, Hill $n_{\mathrm{H}}$, amplitude, MFE $Z$, $\Delta P_{\mathrm{RBS}}$, $\Delta\Delta G$,
<br>
- those scalars *are* the melting phenotype,
<br>
- StratifiedGroupKFold AUC collapsed to $\approx 0.19$ (no transferable detector).

<br>

**Non-circular principle (this model)**:
<br>
<br>
- $X$ uses only **static 37 °C** biophysics, composition, intensive $k$-mer frequencies, and SD–AUG geometry,
<br>
- melting scalars remain on the fused table and are scored **after** $\hat{y}$ as post-hoc gates.

---

## 2. Labelled corpus

### 2.1 Classes

| Class | Source | Label $y$ | $N$ |
|-------|--------|-----------|-----|
| Positive RNATs | Rfam prokaryotic thermoswitch families | $1$ | $1198$ |
| Negative controls | NCBI RefSeq housekeeping 5′ UTRs | $0$ | $1198$ |

<br>

**Positive curation (summary)**:
<br>
<br>
- Rfam thermoswitch families → Entrez FASTA,
<br>
- CD-HIT at **80%** identity → $1198$ representatives.

<br>

**Negative curation (ENN historical / production fused panel)**:
<br>
<br>
- RefSeq 5′ UTR candidates,
<br>
- Infernal `cmscan` decontamination,
<br>
- length / %GC Hungarian match with $|\Delta L| \le 40$, $|\Delta GC| \le 0.05$ (CDS-proximal truncate),
<br>
- balanced table $N = 2396$.

<br>

**RUS ablation (parallel sidecars, not the default fused path)**:
<br>
<br>
- same decontaminated pool,
<br>
- **ENN skipped**; `RandomUnderSampler` → `rus_cleaned.*`,
<br>
- rematch with $|\Delta GC| \le 0.10$,
<br>
- artifacts: `fused_features_refseq_dynamic_rus.csv`, `rf_*_rus.*`.

<br>

**Primary fused training table**:
<br>
<br>
- `data/processed/fused_features_refseq_dynamic.csv`
<br>
- ViennaRNA + NUPACK physics joined to matched sequences.

---

## 3. Feature representation $X$

Let each sequence be $s \in \{A,U,G,C\}^L$. Features are **intensive** where relevant so length does not re-enter as a raw count proxy.

### 3.1 Static 37 °C biophysics (Vienna / NUPACK)

Computed at $37^\circ\mathrm{C}$ (and related static ensemble summaries), including:

<br>

- $E_{\mathrm{MFE}} / L$ (`*_MFE_per_nt`),
<br>
- ensemble diversity, mean positional entropy,
<br>
- max stem / max loop lengths,
<br>
- $P_{\mathrm{paired,RBS}}(37^\circ\mathrm{C})$ (`P_paired_RBS_37`).

<br>

**MFE intensity**:
<br>
<br>
$$
\mathrm{MFE\_per\_nt} = \frac{E_{\mathrm{MFE}}(s;\,37^\circ\mathrm{C})}{L}.
$$

### 3.2 Composition

<br>

- sequence length $L$,
<br>
- %GC (Vienna column retained; duplicate NUPACK GC dropped when redundant),
<br>
- $P_{\mathrm{paired,RBS}}(37^\circ\mathrm{C})$.

### 3.3 Intensive $k$-mer frequencies

For $k \in \{2,3\}$ over alphabet $\{A,U,G,C\}$:

<br>

$$
f_m(s) = \frac{\#\{i : s_{i:i+k-1} = m\}}{L - k + 1},
\qquad m \in \Sigma^k,
$$

<br>

with $f_m = 0$ if $L < k$.

<br>

**Column counts**:
<br>
<br>
- $16$ dinucleotides (`dinuc_*`),
<br>
- $64$ trinucleotides (`trinuc_*`).

### 3.4 SD–AUG geometry

Let $\mathrm{SD}$ be the detected Shine–Dalgarno window and $\mathrm{AUG}$ the initiator codon.

<br>

**Encoding (never drop a row for missing AUG)**:
<br>
<br>
- if AUG found downstream of SD: `sd_aug_spacing` $= \mathrm{AUG\_start} - \mathrm{SD\_end}$, `sd_aug_missing` $= 0$,
<br>
- if missing: `sd_aug_spacing` $= -1$ (sentinel), `sd_aug_missing` $= 1$.

<br>

RefSeq negatives lack AUG more often than Rfam (documented in App 1 / feature log). Dropping those rows would unbalance classes; sentinels keep $N = 2396$.

### 3.5 Explicit exclusions from $X$ (circular scalars)

The following are **forbidden** as RF inputs and reserved for post-hoc scoring:

<br>

- `viennarna_Tm`, `nupack_Tm`,
<br>
- `*_hill_coeff`, `*_amplitude`,
<br>
- `viennarna_mfe_zscore`,
<br>
- `viennarna_delta_P_RBS`,
<br>
- `viennarna_delta_delta_G`.

### 3.6 Realised matrix size

From `rf_noncircular_feature_log.json` / diagnostics:

<br>

- $n = 2396$ rows ($1198$ pos / $1198$ neg),
<br>
- $p = 92$ columns in $X$,
<br>
- physics NaNs may drop a row; SD–AUG sentinels never do.

<br>

**Block structure of $X$** (for grouped permutation):

| Block | Approx. width |
|-------|---------------|
| Static 37 °C biophysics | $7$ |
| Composition | $3$ |
| Dinucleotides | $16$ |
| Trinucleotides | $64$ |
| SD–AUG | $2$ |
| **Total** | **$92$** |

---

## 4. Random Forest model

### 4.1 Ensemble definition

A Random Forest is a bagged ensemble of $T$ classification trees $\{h_t\}_{t=1}^{T}$.

<br>

**Bootstrap aggregation**:
<br>
<br>
- each tree $h_t$ is grown on a bootstrap sample of the training rows,
<br>
- at each split, a random subset of features is considered (sklearn default `max_features="sqrt"` for classification).

<br>

**Gini impurity** at a node with class proportions $p_c$:
<br>
<br>
$$
G = 1 - \sum_{c \in \{0,1\}} p_c^2.
$$

<br>

A split is chosen to minimise the weighted child Gini (sklearn `criterion="gini"`, default).

<br>

**Majority vote / soft aggregation**:
<br>
<br>
$$
\hat{y}(x) = \frac{1}{T} \sum_{t=1}^{T} h_t(x) \in [0,1],
$$

<br>

where $h_t(x)$ is the tree’s estimated class-$1$ probability (fraction of training labels in the leaf). Hard call (not used as a wet-lab gate):

<br>

$$
\hat{c}(x) = \mathbf{1}\{\hat{y}(x) \ge 0.5\}.
$$

### 4.2 scikit-learn parameters (production non-circular RF)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Class | `sklearn.ensemble.RandomForestClassifier` | |
| `n_estimators` ($T$) | **$200$** | |
| `max_depth` | **`None`** | unconstrained depth |
| `criterion` | `"gini"` | sklearn default |
| `max_features` | `"sqrt"` | sklearn classification default |
| `bootstrap` | `True` | default |
| `random_state` | **$42$** | |
| `n_jobs` | **$-1$** | all cores |
| `class_weight` | `None` | balanced by construction ($1{:}1$) |

<br>

**Fit call (conceptual)**:
<br>
<br>
```text
RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1).fit(X, y)
```

<br>

**Persistence**:
<br>
<br>
- model + feature list → `rf_thermoswitch_noncircular.joblib`,
<br>
- JSON sidecar records `feature_set: "noncircular"`, $n$, circular exclusions.

<br>

**Not the honesty ladder**: a separate shallow path in `thermo_classifier.py` uses $T=300$, `max_depth=4` (monotonic XGB / legacy paths). Do **not** confuse that with the non-circular RF.

### 4.3 Schematic train / test figure vs honesty CV

Thesis figures may show a **40% / 60%** train–test cartoon for pedagogy.

<br>

**What the code actually uses for honesty metrics**:
<br>
<br>
- **full-panel fit** for the deployed joblib,
<br>
- **5-fold StratifiedGroupKFold** for out-of-family AUC and OOF $\hat{y}$ (below).

<br>

Do not report the cartoon split as the CV protocol unless a separate holdout script is cited.

---

## 5. Evaluation protocol (honesty)

### 5.1 Grouping key

Each row carries a group id:

<br>

- Rfam positives: `rfam_acc` (family accession),
<br>
- RefSeq negatives: `REFSEQ:{assembly}` (or equivalent assembly-scoped id).

### 5.2 Cross-validation designs

| Protocol | Class | Folds | Role |
|----------|-------|-------|------|
| StratifiedKFold | `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` | $5$ | **leaky** upper bound (family motifs can appear in train and test) |
| StratifiedGroupKFold | `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` | $5$ | **honesty** — no group shared across train/test |

<br>

**Primary metric**: ROC-AUC of $\hat{y}$ vs $y$.

<br>

**Reported non-circular results** (`rf_noncircular_diagnostics.json`):
<br>
<br>
- Stratified (leaky) AUC $\approx \mathbf{0.97}$,
<br>
- StratifiedGroupKFold AUC $\approx \mathbf{0.28}$,
<br>
- length-alone control AUC $\approx 0.20$ (length shortcut removed),
<br>
- historical circular 20-col GroupKFold AUC $\approx 0.19$.

<br>

**Interpretation**:
<br>
<br>
- $0.97$ is family-shaped $k$-mer memory on a random split,
<br>
- $0.28$ is near-chance out of family — ranking aid, not a transferable detector.

### 5.3 Attribution: grouped permutation importance

Do **not** use Gini/MDI for scientific attribution (trinucleotides fragment importance).

<br>

**Procedure**:
<br>
<br>
- fit RF on the full panel,
<br>
- permute feature **blocks**,
<br>
- report mean ROC-AUC drop.

<br>

**Observed block drops** (in-sample forest; see update note):
<br>
<br>
- trinucleotides $\approx 0.34$ AUC drop,
<br>
- dinucleotides $\approx 0.004$,
<br>
- static physics / composition / SD–AUG $\approx 0$.

---

## 6. Post-hoc scoring (not in $X$)

Out-of-fold probabilities from StratifiedGroupKFold are binned, then melting scalars filter rows.

### 6.1 Confidence bins

| Bin | Rule |
|-----|------|
| High | $\hat{y} \ge 0.80$ |
| Mid | $0.40 < \hat{y} < 0.60$ |
| Low | $\hat{y} \le 0.20$ |
| Other | remainder |

<br>

On the labelled panel, the high bin is tiny ($n \approx 6$) under GroupKFold — another view of weak transfer.

### 6.2 Melting / structural gates (panel diagnostics)

| Gate | Rule |
|------|------|
| Heat-shock $T_m$ | $T_m \in [42, 45]^\circ\mathrm{C}$ |
| Hill (numeric) | $n_{\mathrm{H}} > 1.0$ |
| Hill snap (visual) | $n_{\mathrm{H}} > 1.5$ |
| Amplitude | $\Delta\theta \ge 0.50$ |
| Baseline lock | $P_{\mathrm{open}}^{\mathrm{RBS}}(37^\circ\mathrm{C}) \le 0.20$ |
| MFE $Z$ | $Z \le -2$ |
| $\Delta P_{\mathrm{RBS}}$ | $\Delta P_{\mathrm{RBS}} > 0$ |

<br>

**Visual checklist intersection (all four of snap, $T_m$ window, amplitude, baseline)**: historically **$0 / 2396$** on the labelled panel.

### 6.3 Cross-engine consensus

Panel-wide Spearman between Vienna and NUPACK ($T_m$, Hill, amplitude) is near zero on this corpus; high-bin Spearman requires $N \ge 25$ and is underpowered when the high bin has $N < 25$.

### 6.4 Relation to EVA yield (locked, separate)

EVA stream triage does **not** use RF $\hat{y}$. Locked yield gate:

<br>

$$
Z \le -2
\quad\wedge\quad
\Delta P_{\mathrm{RBS}} > 0
\quad\wedge\quad
E_{\mathrm{Rfam}} > 10^{-3}.
$$

---

## 7. Software stack (minimum)

| Component | Spec |
|-----------|------|
| Python | $\ge 3.11$ (project `requires-python`) |
| scikit-learn | `RandomForestClassifier`, `StratifiedGroupKFold`, `StratifiedKFold` |
| pandas / numpy | matrix assembly |
| scipy | post-hoc MW / KS / Spearman; Hill fits elsewhere |
| ViennaRNA / NUPACK | static + dynamic physics on the fused panel |
| joblib | model persistence |

---

## 8. Reproduction recipe

From the repository root:

```bash
# Optional: refresh P_open / intensive columns if needed
PYTHONPATH=src python src/thermo_sim/enrich_dynamic_features.py --p-open-only --workers 4

# Train non-circular RF (default feature_set path via intensive=True)
PYTHONPATH=src python src/thermo_sim/thermo_classifier.py train \
  --fused-csv data/processed/fused_features_refseq_dynamic.csv \
  --model-path data/processed/models/rf_thermoswitch_noncircular.joblib

# Post-hoc bins, gates, consensus
PYTHONPATH=src python src/thermo_sim/thermo_classifier.py posthoc \
  --fused-csv data/processed/fused_features_refseq_dynamic.csv
```

<br>

**Expected artifacts**:

| Path | Contents |
|------|----------|
| `data/processed/models/rf_thermoswitch_noncircular.joblib` | Fitted forest + feature list |
| `data/processed/models/rf_thermoswitch_noncircular.json` | Sidecar metadata |
| `data/processed/rf_noncircular_feature_log.json` | $X$ columns, AUG-missing rates |
| `data/processed/rf_noncircular_diagnostics.json` | Stratified vs GroupKFold AUC, permutation |
| `data/processed/rf_posthoc_report.json` | Bins, gates, Spearman, checklist |

<br>

**Interactive inspection**:
<br>
<br>
- `uv run marimo edit notebooks/06_classifier_architecture_ladder.py`
<br>
- `uv run marimo edit notebooks/07_noncircular_rf_results.py`
<br>
- `uv run marimo edit notebooks/thesis_figures.py` (cohort map + RF ensemble cartoon + parameter card)

---

## 9. Concise Methods paragraph (paste-ready)

> We trained a scikit-learn Random Forest classifier ($T = 200$ trees, unconstrained depth, Gini splits, `random_state=42`) on a length/%GC-matched panel of $N = 2396$ sequences ($1198$ Rfam positives, $1198$ RefSeq 5′ UTR negatives). The feature matrix comprised $92$ non-circular columns: static $37^\circ\mathrm{C}$ Vienna/NUPACK biophysics, length and %GC, RBS pairing probability at $37^\circ\mathrm{C}$, intensive dinucleotide and trinucleotide frequencies, and SD–AUG spacing with a missing-AUG sentinel. Melting scalars ($T_m$, Hill coefficient, amplitude, MFE $Z$, $\Delta P_{\mathrm{RBS}}$, $\Delta\Delta G$) were excluded from $X$ and reserved for post-hoc gates. Honesty was assessed with $5$-fold `StratifiedGroupKFold` grouped by Rfam accession / RefSeq assembly (out-of-family ROC-AUC $\approx 0.28$), contrasted with leaky stratified splits (AUC $\approx 0.97$). The forest is used as a ranking aid, not as a binary wet-lab gate; EVA candidates continue to be admitted by the locked yield rule $Z \le -2 \wedge \Delta P_{\mathrm{RBS}} > 0 \wedge E_{\mathrm{Rfam}} > 10^{-3}$.

---

## 10. Sources of truth in-repo

| Topic | Location |
|-------|----------|
| Non-circular feature build | `src/thermo_sim/noncircular_features.py` |
| Train / persist RF | `src/thermo_sim/thermo_classifier.py` (`train_random_forest`) |
| Post-hoc bins / gates | `src/thermo_sim/rf_posthoc.py` |
| Architecture ladder narrative | `notebooks/06_classifier_architecture_ladder.py` |
| Results tables | `notebooks/07_noncircular_rf_results.py` |
| Model update summary | `notebooks/07_noncircular_rf_model_update.md` |
| Corpus matching / RUS | `README.md` §1; `knn_undersample.py --skip-enn` |
