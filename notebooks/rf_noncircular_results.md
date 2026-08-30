# Results: Non-Circular Random Forest for RNA Thermoswitch Ranking

**Companion to:** [`rf_noncircular_methodology.md`](rf_noncircular_methodology.md)  
**Panel:** $N = 2396$ ($1198$ Rfam / $1198$ RefSeq), $p = 92$ non-circular features  
**Model:** `rf_thermoswitch_noncircular.joblib` ($T = 200$, unconstrained depth)  
**Primary sources:** `rf_noncircular_diagnostics.json`, `rf_posthoc_report.json`  
**Ablation:** RUS sidecars `*_rus.json` (ENN skipped)

> After removing melting scalars from $X$, does the forest learn transferable thermoswitch physics—or only Rfam-family $k$-mer memory?

<br>

**Verdict in one line**:
<br>
<br>
- Non-circularisation **kills the circular shortcut** but **does not** produce an out-of-family detector.
<br>
- Stratified AUC $\approx 0.97$ is **family leakage**; StratifiedGroupKFold AUC $\approx 0.28$ is **near chance**.
<br>
- Use $\hat{y}$ as a **ranking aid**, never as a wet-lab gate.

---

## 1. Experimental readout definition

All honesty numbers below use:

<br>

- **Leaky upper bound:** 5-fold `StratifiedKFold` (family motifs may train and test),
<br>
- **Honesty metric:** 5-fold `StratifiedGroupKFold` on `rfam_acc` / `REFSEQ:{assembly}`,
<br>
- **Score:** ROC-AUC of soft ensemble probability $\hat{y} \in [0,1]$,
<br>
- **Chance:** $0.5$ for balanced $y$.

<br>

**Controls / baselines on the same panel**:
<br>
<br>
- length-alone RF (composition shortcut check),
<br>
- historical circular 20-column RF (GroupKFold $\approx 0.19$).

---

## 2. Discrimination performance

### 2.1 Primary non-circular RF (ENN-matched panel)

| Protocol | ROC-AUC | Accuracy | Read as |
|----------|--------:|---------:|---------|
| Length alone (Stratified) | $0.196$ | $0.273$ | Length shortcut is gone |
| Non-circular Stratified (leaky) | $\mathbf{0.966}$ | $0.902$ | Family-shaped $k$-mer memory |
| Non-circular StratifiedGroupKFold | $\mathbf{0.277}$ | $0.470$ | Near-chance out of family |
| Circular 20-col GroupKFold (history) | $\approx 0.19$ | — | Circular phenotype in $X$ |

<br>
Source: `data/processed/rf_noncircular_diagnostics.json` (rounded).

<br>

**What a machine-learning biotechnologist should conclude**:
<br>
<br>
- Matching length/%GC worked: length alone cannot separate classes (AUC $< 0.20$).
<br>
- Random-split $0.97$ is **not** evidence of a thermoswitch detector; it is consistent with memorising family-correlated trinucleotides when the family is seen at train time.
<br>
- GroupKFold $0.28$ is a **small lift** over the circular history ($0.19$), not a deployable classifier ($\Delta\mathrm{AUC} \approx +0.09$ vs circular; still $\ll 0.70$–$0.80$ useful diagnostic territory).
<br>
- Accuracy $0.47$ under GroupKFold is compatible with near-random hard calls on a balanced set.

### 2.2 Negative-panel ablation: ENN vs RUS

Same non-circular $X$ and RF hyperparameters; only negative construction changes (ENN cleaning vs RUS-only).

| Panel | Stratified AUC | GroupKFold AUC | Top perm. block ($\Delta$AUC) |
|-------|---------------:|---------------:|-------------------------------|
| ENN (historical fused) | $0.966$ | $0.277$ | Trinucleotides $\approx 0.338$ |
| RUS (no ENN) | $0.966$ | $0.271$ | Trinucleotides $\approx 0.334$ |

<br>
Source: `logs/rus_panel_rebuild.md`, `rf_noncircular_diagnostics_rus.json`.

<br>

**Ablation read**:
<br>
<br>
- Switching negatives from ENN→RUS does **not** rescue transfer,
<br>
- stratified inflation and GroupKFold collapse are **stable** to that design choice,
<br>
- failure mode is therefore not an ENN-specific artifact of the negative pool.

---

## 3. What the forest actually uses (attribution)

Grouped permutation importance on a **full-fit** forest (in-sample baseline AUC $= 1.0$; $5$ repeats).

<br>

**These numbers explain the overfit model’s crutches—not GroupKFold transfer.**

| Feature block | $n$ features | Mean AUC drop | Std |
|---------------|-------------:|--------------:|----:|
| Trinucleotides | $64$ | $\mathbf{0.338}$ | $0.010$ |
| Dinucleotides | $16$ | $0.004$ | $0.0004$ |
| Static 37 °C biophysics | $7$ | $\approx 0$ | $\approx 0$ |
| Composition ($L$, %GC, $P_{\mathrm{paired},37}$) | $3$ | $0$ | $0$ |
| SD–AUG (spacing + missing) | $2$ | $0$ | $0$ |

<br>

**Biotech / ML synthesis**:
<br>
<br>
- The forest’s discriminative mass lives in the **64-mer block**,
<br>
- static thermodynamics at $37^\circ\mathrm{C}$ contribute essentially **nothing** to in-sample AUC under permutation,
<br>
- that pattern **predicts** Stratified $0.97$ / GroupKFold $0.28$: motifs separate families when shared across folds and fail when families are held out.

<br>

Do **not** report Gini/MDI as scientific importance for this $X$ (correlated trinucleotides fragment impurity).

---

## 4. Calibration of confidence under honesty CV

Out-of-fold $\hat{y}$ from StratifiedGroupKFold (`rf_posthoc_report.json`):

| Bin | Rule | $n$ | Fraction of panel |
|-----|------|----:|------------------:|
| High | $\hat{y} \ge 0.80$ | $\mathbf{6}$ | $0.25\%$ |
| Mid | $0.40 < \hat{y} < 0.60$ | $297$ | $12.4\%$ |
| Low | $\hat{y} \le 0.20$ | $1504$ | $62.8\%$ |
| Other | remainder | $589$ | $24.6\%$ |

<br>

**Operational consequence**:
<br>
<br>
- Almost nobody is high-confidence **out of family**,
<br>
- a threshold such as $\hat{y} \ge 0.80$ cannot be a discovery gate (support $n=6$),
<br>
- mid-bin mass is modest; the model mostly assigns low probability under group holdout.

---

## 5. Post-hoc melting phenotype (not RF inputs)

Melting scalars were **excluded from $X$**. They are scored on the labelled panel as diagnostics.

### 5.1 Single-gate pass counts ($N = 2396$)

| Gate | Rule | $n$ pass |
|------|------|--------:|
| $\Delta P_{\mathrm{RBS}} > 0$ | RBS opens with heat (Vienna) | $2294$ |
| Vienna $n_{\mathrm{H}} > 1.0$ | cooperative Hill | $2377$ |
| $T_m \in [42,45]^\circ\mathrm{C}$ | heat-shock window | $168$ |
| $Z \le -2$ | unusually stable MFE vs composition null | $1128$ |

<br>

**Read**:
<br>
<br>
- “Has a Hill slope” is almost universal on this panel,
<br>
- sitting in the **narrow heat-shock $T_m$ window** is rare ($7.0\%$),
<br>
- gates are **not stacked** in the table above (independent counts).

### 5.2 Visual four-signature checklist

| Signature | Pass rule | $n$ pass |
|-----------|-----------|--------:|
| Sigmoidal snap | $n_{\mathrm{H}} > 1.5$ | $2355$ |
| Inflection | $T_m \in [42,45]^\circ\mathrm{C}$ | $168$ |
| Dynamic range | $\Delta\theta \ge 0.50$ | $699$ |
| Baseline repression | $P_{\mathrm{open}}^{\mathrm{RBS}}(37^\circ\mathrm{C}) \le 0.20$ | $75$ |
| **All four** | intersection | $\mathbf{0}$ |

<br>

**Result**:
<br>
<br>
- Steep Hills are common; locked basal RBS + heat-shock $T_m$ + large stroke **do not co-occur** on this labelled panel,
<br>
- $\mathrm{all\_four\_pass} = 0$ also holds on the **RUS** fused panel.

### 5.3 Cross-engine consensus (Vienna vs NUPACK)

Panel-wide Spearman ($n = 2396$; primary metric):

| Pair | $r_s$ | $p$ |
|------|------:|----:|
| $T_m$ | $0.035$ | $0.083$ |
| Hill $n_{\mathrm{H}}$ | $0.037$ | $0.072$ |
| Amplitude | $0.033$ | $0.103$ |

<br>

**High-bin Spearman**: **not reported** ($n_{\mathrm{high}} = 6 < 25$; underpowered by protocol).

<br>

**RUS panel $T_m$ $r_s$** $\approx 0.019$ — still near zero.

<br>

**Interpretation**:
<br>
<br>
- The two engines **barely rank-agree** on melting scalars for this corpus,
<br>
- engine concordance cannot rescue RF transfer; it is an orthogonal reliability check.

### 5.4 High vs low $\hat{y}$ bins (exploratory only)

Mann–Whitney / KS on Vienna melting columns, high ($n=6$) vs low ($n=1504$):

| Scalar | MW $p$ | KS $p$ | Stable phenotype? |
|--------|-------:|-------:|-------------------|
| $T_m$ | $0.007$ | $0.002$ | **No** — $n_{\mathrm{high}}=6$ |
| Hill | $0.74$ | $0.35$ | No |
| Amplitude | $0.56$ | $0.86$ | No |
| MFE $Z$ | $0.64$ | $0.72$ | No |
| $\Delta P_{\mathrm{RBS}}$ | $0.55$ | $0.44$ | No |

<br>

Treat $T_m$ $p$-values as **hypothesis-generating only**. With six high scores, effect sizes are not operationally usable.

---

## 6. Placement in the broader pipeline

| Role | Finding |
|------|---------|
| RF as detector | **Fails** honesty CV (AUC $0.28$) |
| RF as ranker | **Usable with caveats** — non-tautological score vs melting |
| Post-hoc melting gates | Describe panel phenotype; do not validate $\hat{y}$ |
| EVA admission | **Independent** locked yield: $Z\le-2 \wedge \Delta P_{\mathrm{RBS}}>0 \wedge E_{\mathrm{Rfam}}>10^{-3}$ → $105/2000$ |

<br>

**Expert framing**:
<br>
<br>
- This is a classic **distribution-shift / family-leakage** result in biological sequence ML,
<br>
- removing label-leaking phenotype features was necessary scientific hygiene,
<br>
- the remaining signal is **sequence-family specific**, not a global thermoswitch physics rule in the current $X$.

---

## 7. Limitations (explicit)

<br>

- GroupKFold groups RefSeq by assembly; residual homology within assemblies can still leak.
<br>
- Full-fit permutation importance overstates what transfers.
<br>
- High-bin analyses are underpowered by construction under honesty CV.
<br>
- Vienna–NUPACK disagreement limits phenotype filters that assume engine concordance.
<br>
- EVA FASTA stream triage still lacks Hill/$T_m$ until `thermo_batch` on generated sequences.

---

## 8. Concise Results paragraph (paste-ready)

> On the length/%GC-matched panel ($N=2396$), a non-circular Random Forest ($T=200$, 92 features excluding melting scalars) achieved stratified ROC-AUC $0.966$ but StratifiedGroupKFold AUC $0.277$, only marginally above the historical circular model ($\approx 0.19$) and far below a useful diagnostic threshold. Length-alone AUC was $0.196$, confirming matching removed the trivial length cue. Grouped permutation importance localised in-sample discrimination to trinucleotide frequencies (mean AUC drop $0.338$), while static $37^\circ\mathrm{C}$ biophysics and SD–AUG features contributed negligibly. Out-of-family high-confidence scores ($\hat{y}\ge 0.80$) numbered only six. Post-hoc visual checklist intersection (Hill snap, heat-shock $T_m$, amplitude, basal RBS lock) was empty ($0/2396$). Vienna–NUPACK Spearman correlations for $T_m$, Hill, and amplitude were $\approx 0.03$–$0.04$ (non-significant). An RUS negative-panel ablation reproduced the same stratified/GroupKFold split ($0.966$ / $0.271$). We therefore treat the forest as a non-circular ranking aid, not a transferable thermoswitch classifier.

---

## 9. Artifact index

| File | Role |
|------|------|
| `data/processed/rf_noncircular_diagnostics.json` | Stratified / GroupKFold AUC + permutation |
| `data/processed/rf_posthoc_report.json` | Bins, gates, Spearman, MW/KS, checklist |
| `data/processed/rf_noncircular_diagnostics_rus.json` | RUS ablation AUCs |
| `data/processed/rf_posthoc_report_rus.json` | RUS post-hoc |
| `logs/rus_panel_rebuild.md` | ENN vs RUS summary table |
| `notebooks/07_noncircular_rf_results.py` | Executable result tables |
| `notebooks/figures/07_classifier/` | Permutation / checklist figures |

<br>

**Methods twin:** [`rf_noncircular_methodology.md`](rf_noncircular_methodology.md)
