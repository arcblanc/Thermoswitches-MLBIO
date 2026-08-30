# Evaluating EVA as a Generator of Candidate RNA Thermoswitches

**Source:** Marimo App 8 — `notebooks/08_eva_denovo_biophysical_characterization.py`  
**Cohort:** 105 yield-gated EVA sequences (31 pilot + 74 stream) vs 1,198 Rfam / 1,198 RefSeq  
**Updates:** two-tier Neupert checklist, recalibrated lead score, top-3 deep audits, §13 Lead-315 G–C clamp PoC  
**Artifacts:** `data/processed/eva_denovo_checklist.json`, `data/processed/eva_characterization/top3_deep_audit.json`, `data/processed/eva_characterization/poc_315_gc_clamp.json`, `notebooks/figures/08_eva_denovo/`

---

## 1. Research question

> Can EVA + yield triage conjure heat-inducible thermoswitches that are *biologically plausible* (Tier 1) and *Neupert-ready* for *E. coli* heat-shock induction (Tier 2)—or only structured RNA that still needs rational tuning?

---

## 2. Evaluation standard (literature-grounded)

### Tier 1 — Biological plausibility

- $n_{\mathrm{H}} > 1.2$
- $T_{\mathrm{m}} \in [35, 55]^\circ\mathrm{C}$
- $\Delta\theta \ge 0.20$
- $P_{\mathrm{open}}^{37^\circ\mathrm{C}} \le 0.35$

**Purpose:** non-zero survival baseline for natural Rfam; ask whether EVA samples viable folds.

### Tier 2 — Neupert 2008 / synbio heat-inducible spec

- $n_{\mathrm{H}} > 1.5$
- $T_{\mathrm{m}} \in [42, 45]^\circ\mathrm{C}$
- $\Delta\theta \ge 0.40$ (ideal $\rightarrow 0.50$)
- $P_{\mathrm{open}}^{37^\circ\mathrm{C}} \le 0.20$

**Purpose:** turnkey controllers for heat-shock induction without secondary redesign.

**Neupert et al. (2008):** primary Tier 2 math — heat-*inducible* RBS unmasking (FourU class).  
**Hoynes-O’Connor et al. (2015):** method template — dual controls, orthogonality, mechanism rescue (heat-*repressible* RNase E class); maps to FourU overlay + in silico disrupt/rescue.

---

## 3. Two-tier checklist results

| Cohort | $N$ | Tier 1 all-4 | Tier 1 frac | Tier 2 all-4 | Tier 2 frac |
|--------|----:|-------------:|------------:|-------------:|------------:|
| Natural Rfam | 1198 | **29** | 2.4% | **0** | 0% |
| RefSeq negatives | 1198 | 32 | 2.7% | **0** | 0% |
| EVA pilot | 31 | **1** | 3.2% | **0** | 0% |
| EVA stream | 74 | **1** | 1.4% | **0** | 0% |

**Finding.**  
Natural thermoswitches and EVA both show **non-zero Tier 1** survival.  
Both collapse to **zero** under simultaneous Tier 2 Neupert filters.  
The strict filter is therefore not an EVA-only failure — it is a hard joint phenotype that even Rfam rarely meets in this panel.

---

## 4. Recalibrated ranking objective

$$
\mathrm{Score}
= w_1 \min(n_{\mathrm{H}}, 3)
+ w_2 |\Delta\theta|
- w_3 |T_{\mathrm{m}}-43.5|
- w_4 P_{\mathrm{open}}^{37}
- w_5 |\Delta T_{\mathrm{m}}^{\mathrm{Vienna-NP}}|
- \mathrm{Penalty}_{\mathrm{stroke}}
$$

with $\mathrm{Penalty}_{\mathrm{stroke}} = 12 \cdot \max(0,\ 0.25-|\Delta\theta|)$, $w_4=6$, $w_2=2$.

**Effect vs prior $n_H=20$ ranking:** top ranks now prefer measurable stroke and heat-shock-proximal $T_m$, not optimiser ceilings.

### Re-ranked top 10 (excerpt)

| Rank | ID | Score | $n_H$ | $T_m$ | $\Delta\theta$ | $P_{37}$ |
|-----:|----|------:|------:|------:|---------------:|---------:|
| 1 | eva_sample_315 | +1.09 | 8.08 | 46.1 | **0.46** | 0.30 |
| 2 | eva_sample_1914 | −0.04 | 8.71 | **44.1** | −0.03 | **0.03** |
| 3 | eva_sample_505 | −0.66 | 6.04 | 38.6 | 0.36 | 0.40 |

Lead **315** is the only top entry with Tier-1-grade stroke near the Tier-2 floor.  
**1914** is tightly locked but essentially stroke-less (fails the new amplitude logic despite excellent leak).  
**505** has moderate stroke but $T_m$ below the Neupert window and a leaky baseline.

---

## 5. Deep audits on re-ranked top 3

### 5.1 Positional unpairing heatmaps

| Lead | SD window | $P_{\mathrm{unpaired}}^{\mathrm{SD}}(37)$ | $P_{\mathrm{unpaired}}^{\mathrm{SD}}(43)$ | Global mean (43) |
|------|-----------|------------------------------------------:|------------------------------------------:|-----------------:|
| 315 | nt 400–405 | 0.30 | 0.39 | 0.64 |
| 1914 | nt 42–45 | 0.030 | 0.025 | 0.37 |
| 505 | nt 415–420 | 0.40 | 0.49 | 0.51 |

**Finding.**  
None of the top 3 shows a clean “SD lights up, scaffold stays dark” Neupert signature at 42–45 °C.  
- **315:** SD opens modestly with heat, but the transcript is already globally unpaired (~0.63).  
- **1914:** SD stays locked (good basal) and does **not** open at 43 °C (no local switch).  
- **505:** SD opens somewhat, but baseline is already leaky and global melt is high.

Figures: `notebooks/figures/08_eva_denovo/heatmap_eva_sample_{315,1914,505}.png`

### 5.2 MDS structural embedding (full EVA library)

- $N=105$ MFE structures  
- **105 unique** dot-brackets  
- mean / median tree-edit distance $\approx 367$ / $382$

**Finding.**  
EVA is **not** mode-collapsed at the secondary-structure level after yield triage — topological diversity is high.  
Diversity ≠ Neupert phenotype; many distinct folds still fail Tier 2.

Figure: `notebooks/figures/08_eva_denovo/structure_manifold.png`

### 5.3 Mutational disrupt / rescue (Hoynes-O’Connor-style causality)

| Lead | WT $\Delta P_{43-37}$ | Disrupt $P_{37}$ | Rescue $P_{37}$ | Causal pattern? |
|------|----------------------:|-----------------:|----------------:|-----------------|
| 315 | +0.093 | 0.60 (opened) | 0.81 (worse leak) | Partial: disrupt kills lock; rescue does **not** restore WT |
| 1914 | −0.006 | 0.29 | 0.29 | No: WT has no ON stroke; mutate barely moves curve |
| 505 | +0.083 | 0.24 | 0.37 | Mixed: disrupt lowers leak oddly; rescue ≠ WT snap |

**Finding.**  
Stem edits change RBS exposure, so pairing chemistry matters — but none of the top 3 shows the clean Neupert-style narrative “mismatch destroys heat-shock snap; compensatory rescue restores $42$–$45^\circ\mathrm{C}$ sigmoid.”

Figure: `notebooks/figures/08_eva_denovo/mutation_rescue.png`

---

## 6. Thesis-level verdict

**What holds.**

1. Yield triage + EVA produces over-stabilised, structurally diverse RNA.  
2. Tier 1 proves a non-zero biological baseline (Rfam 29; EVA 2 all-four).  
3. Recalibrated scoring demotes $n_H=20$ / zero-stroke artifacts.  
4. Deep audits supply Hoynes-O’Connor-style mechanism tests in silico.

**What does not hold.**

1. Tier 2 Neupert readiness remains **0 / 105** (and **0 / 1198** Rfam).  
2. Top-3 heatmaps do not show modular RBS-local melting at heat shock.  
3. Disrupt/rescue does not yet certify a finished, causally clean thermometer.

**Balanced claim.**

> EVA is a viable *proposal engine* for heat-inducible thermoswitch discovery when scored with Tier 1 / Tier 2 separation and stroke-aware ranking — but App 8 still shows that generative admission and structural diversity are upstream of Neupert-spec function. The next bottleneck is redesign (or constrained generation) for simultaneous heat-shock $T_m$, $\Delta\theta \ge 0.40$–$0.50$, and $P_{\mathrm{open}}^{37} \le 0.20$, validated by local heatmaps and mutational rescue. A one-sequence scaffold G–C clamp on lead 315 (**Scenario 3**) confirms that instability is distributed, not a single-stem fix.

---

## 7. One-sentence abstract

Under a two-tier Neupert-grounded checklist and a stroke-capped ranking function, EVA yields rare Tier-1 survivors and diverse folds, yet zero Tier-2 turnkey thermoswitches; top-3 audits and a Lead-315 scaffold G–C clamp PoC (Scenario 3) confirm current leads are distributed-instability design intermediates, not finished heat-shock RBS controllers.

---

## 8. Quick reference

| Quantity | Result |
|----------|--------|
| Tier 1 all-4 (Rfam / EVA) | 29 / 2 |
| Tier 2 all-4 (all cohorts) | 0 |
| Score $n_H$ cap / stroke floor | 3.0 / 0.25 |
| Top-1 lead | eva_sample_315 ($T_m\approx46^\circ\mathrm{C}$, $|\Delta\theta|\approx0.46$) |
| Unique MFE topologies (EVA) | 105 / 105 |
| Top-3 clean Neupert heatmap | 0 / 3 |
| Top-3 clean disrupt→rescue restore | 0 / 3 |
| Lead-315 scaffold G–C clamp PoC | **Scenario 3** (scaffold still frays) |

---

## 9. §13 Proof-of-concept: manual scaffold rescue on `eva_sample_315`

Testing a manual 1-sequence rescue on **`eva_sample_315`** compares the positional melting heatmap of the wild-type EVA sequence against an engineered variant where 2–3 base pairs in the non-RBS structural stem are mutated to strong G–C clamps.

**Edits applied (farthest non-SD A–U pairs in the MFE):** nt15–31 U–A→C–G, nt17–29 A–U→G–C, nt18–28 U–A→C–G (0-based: 14/30, 16/28, 17/27).  
**Figure:** `notebooks/figures/08_eva_denovo/poc_315_before_after_heatmap.png`  
**Metrics cache:** `data/processed/eva_characterization/poc_315_gc_clamp.json`

Running this test produces one of three distinct biophysical outcomes:

### The Three Possible Result Scenarios

| Rescue Outcome | Heatmap Visual Readout | Biophysical Metric Shift | Scientific Meaning |
| --- | --- | --- | --- |
| **1. Ideal Modular Rescue** *(Best Case)* | The non-RBS scaffold remains dark/paired at $42\text{--}45^\circ\text{C}$, while only the SD window (nt 400–405) lights up bright yellow/unpaired. | Global mean unpairing drops from **$0.64 \to \le 0.25$**; SD stroke increases from **$0.09 \to \ge 0.40$**. | **Validates the Hybrid Design Hypothesis:** Proves EVA generated a viable functional topology that only lacked scaffold stability, completing the *de novo* design pipeline. |
| **2. Negative Design Trap** *(Misfolded / Locked)* | The entire sequence remains dark/paired across all temperatures ($30\text{--}50^\circ\text{C}$); the SD never opens. | $P_{\text{open}}^{\text{SD}}(43^\circ\text{C})$ collapses to $\le 0.10$; dynamic stroke $\Delta\theta \approx 0$. | **Exposes Alternative Base-Pair Trapping:** The newly introduced G and C bases accidentally paired with the Shine–Dalgarno sequence, creating an unintended, permanently locked ground state. |
| **3. Partial / Ineffective Rescue** *(Scaffold Still Frays)* | The heatmap shows slightly delayed melting, but the entire scaffold still lights up above $45^\circ\text{C}$. | Global unpairing stays high ($\ge 0.50$); $T_{\text{m}}$ shifts upward by only $1\text{--}2^\circ\text{C}$. | **Reveals Distributed Structural Instability:** Instability in EVA’s generated fold is not isolated to one stem, but distributed across multiple internal loops, requiring multi-stem combinatorial optimization. |

### What this run showed (Scenario 3)

| Metric | Wild-type | +3 G–C clamps |
|--------|----------:|--------------:|
| Global $P_{\mathrm{unpaired}}$ @ 43 °C | 0.64 | **0.65** (no drop) |
| Scaffold (non-SD) @ 43 °C | 0.65 | 0.65 |
| SD $P_{\mathrm{open}}$ @ 37 °C | 0.30 | **0.73** (leakier) |
| SD $P_{\mathrm{open}}$ @ 43 °C | 0.39 | 0.81 |
| SD stroke $\Delta\theta$ (43−37) | 0.09 | **0.08** (unchanged) |

**Verdict: Scenario 3 — Partial / Ineffective Rescue**, with a leakiness side-effect.

- The Before vs After heatmaps both stay bright across the scaffold at 42–45 °C; clamping three distant A–U stems did **not** darken the non-RBS body.
- SD stroke stayed ~0.09 (far below the ≥0.40 Neupert target).
- Clamps redistributed the ensemble toward a **leakier** SD at 37 °C (0.30 → 0.73) without creating a modular ON switch — a mild negative-design side-effect on top of failed scaffold rescue.

### What Each Result Means for Your Thesis

**If You Get Scenario 1 (Ideal Rescue):**

* **The Claim:** You demonstrate a complete end-to-end synthetic biology proof of concept: generative AI (EVA) proposes the complex fold topology, and targeted rational bioengineering converts it into a Tier-2 Neupert-compliant switch.
* **Thesis Impact:** Provides a definitive "Before vs. After" figure showing global denaturation resolved into clean, localized RBS opening.

**If You Get Scenario 2 (Negative Design Trap):**

* **The Claim:** You demonstrate the classic **negative design bottleneck** in RNA synthetic biology—strengthening one stem often creates unintended alternative base pairings that destroy switching dynamics.
* **Thesis Impact:** Justifies why simple manual mutations are insufficient and proves that automated multi-state partition function optimization (like simulated annealing or `RNAinverse`) is necessary for reliable downstream design.

**If You Get Scenario 3 (Partial Rescue):** ← *this run*

* **The Claim:** You prove that EVA's generative prior sampled a globally diffuse folding ensemble rather than an isolated modular cassette.
* **Thesis Impact:** Establishes the exact design boundary for future generative RNA models, showing they must be trained with explicit structural modularity constraints rather than sequence-level thermodynamic loss functions alone. Manual 2–3 pair G–C clamps are insufficient; multi-stem / multi-state redesign is the next step.
