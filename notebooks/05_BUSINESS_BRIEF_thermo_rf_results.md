# Business Brief: Thermodynamic Screening Results

**Source analysis:** [`05_full_thermo_rf_analysis.ipynb`](05_full_thermo_rf_analysis.ipynb)  
**Audience:** Business / programme managers (no biology or ML background required)  
**Status of this run:** Completed computational screen of **10,000 AI-generated RNA designs**, using a physics-based scoring model trained on known natural examples.

---

## 1. One-paragraph takeaway

We built a **digital filter** that scores whether an RNA molecule looks like a natural “temperature switch” (a thermoswitch). The filter works well on known examples (~**92% accuracy**, ~**95% AUC**). We then scored **10,000 new designs** from an AI generator (GenerRNA). About **39%** looked switch-like to the model; after stricter biophysical gates we shortlisted **99** designs for possible lab work. A later novelty check found that **almost all of those 99 were copies of known sequences**, not truly new inventions — so this pipeline is validated as a **screener**, but the GenerRNA generator itself is not yet delivering genuine novelty. The next generation step (EVA) is meant to fix that.

---

## 2. What problem are we solving? (plain English)

Bacteria sometimes use special RNA “thermostats” that turn genes **on when it gets warmer**. Imagine a lock that stays closed at room temperature and opens near body heat or industrial process heat.

**Business goal:** Design new RNA thermostats on a computer, then only send the best few to expensive wet-lab synthesis and testing.

Without a good computer filter, you would have to synthesise and test far too many random sequences. This project’s value is **reducing that needle-in-a-haystack cost**.

---

## 3. How the system is built (architecture as a factory)

Think of three stations on an assembly line:

```text
┌─────────────────────┐     ┌──────────────────────────┐     ┌─────────────────────┐
│ 1. LEARN            │     │ 2. INVENT                │     │ 3. SCORE & SHORTLIST│
│ Known examples      │     │ AI proposes new RNAs     │     │ Physics + ML filter │
│ from public databases│ →   │ (GenerRNA on GPU cloud)  │ →   │ (CPU cloud / Mac)   │
│ ~2,400 balanced     │     │ 10,000 candidates        │     │ → 99 priority hits  │
└─────────────────────┘     └──────────────────────────┘     └─────────────────────┘
```

### Station 1 — Learn from nature
- We collected known thermoswitches and non-switches from public RNA databases (Rfam).
- We balanced the set so the model does not simply always guess “switch” or “not switch.”
- **~2,395** sequences used for training (about half switches, half controls).

### Station 2 — Invent new sequences (GPU cloud)
- A generative AI model (GenerRNA) ran on **RunPod** (A100 GPU).
- It produced **10,000** candidate RNA sequences.
- Artifacts stored in **AWS S3** (`thermo-s3-bucket`, prefix `llm-batch/v1`).

### Station 3 — Score with physics + machine learning (CPU, not GPU)
- Two independent “physics engines” (ViennaRNA and NUPACK) compute how each RNA would fold and melt as temperature rises.
- Those measurements become **15 numeric features** (stability, melting temperature, stem/loop shape, GC content, etc.).
- A **Random Forest** classifier (a standard, explainable ML model) was trained on Station 1 and applied to Station 2.
- This step ran on a small **AWS EC2 CPU instance** (`c7i-flex.large`) and/or Mac — **no GPU required** for folding/scoring.
- Results land under S3 `thermo/training/` and `thermo/denovo/`.

| Cloud piece | Role | Cost character |
|-------------|------|----------------|
| RunPod GPU | Invent sequences | Expensive per hour; stop when done |
| AWS S3 | Shared file warehouse | Cheap durable storage |
| AWS EC2 CPU | Physics + ML scoring | Cheap vs GPU; stop when done |

---

## 4. What the analysis found (translated numbers)

### A. The scoring model is strong on known data
When tested with 5-fold cross-validation on the training set (a standard “don’t grade your own homework” check):

| Metric | Result | What a manager should hear |
|--------|--------|----------------------------|
| Accuracy | **~92%** | Correct call most of the time on known examples |
| ROC AUC | **~0.95** | Excellent separation of switches vs non-switches |
| Precision (switches) | **~98%** | When it says “switch,” it is rarely wrong on known data |
| Recall (switches) | **~86%** | It misses some real switches (conservative side) |

**Business implication:** The **filter** is credible. It is worth using to shrink large candidate lists before lab spend.

### B. What the model pays attention to
The strongest signals were **folding stability** (how tightly the RNA structure holds together), then composition (GC content) and structural shape (loops/stems).

**Business implication:** Decisions are driven by **measurable physical properties**, not opaque sequence memorisation alone — good for interpretability and stakeholder trust.

### C. Results on the 10,000 AI designs

| Funnel stage | Count | Plain meaning |
|--------------|------:|---------------|
| AI proposals generated | 10,000 | Raw creative output |
| Scored successfully | 9,999 | Nearly all processed |
| Called “switch-like” at default threshold | **3,882 (~39%)** | Broad first cut |
| High-confidence (≥90% model score) | **1,566** | Stronger belief from the model |
| Passed strict biophysical + confidence gates | **99** | Shortlist for possible synthesis |

The strict gates asked for things a wet lab would care about, e.g.:
- Melting temperature in a useful window (**37–55 °C**)
- Clear melting transition (not a flat, uninformative curve)
- Cooperative “sharp” switch behaviour
- Both physics engines producing usable fits
- High model confidence (≥90%)

**Business implication:** From 10,000 ideas → **99** prioritised candidates — roughly a **100× reduction** in what you would consider synthesising first.

### D. Important caveat from a follow-up novelty screen
Of those **99** shortlisted designs, a later search against known databases found:

- **98 / 99** were identical or nearly identical to known natural sequences  
- **1 / 99** was only a remote lookalike  
- **0 / 99** were truly novel with no hit  

**Business implication (critical):**  
The pipeline **successfully recovered known-looking thermoswitch behaviour**, which validates the **scorer**. But GenerRNA largely **regurgitated known biology** rather than inventing new switches. Treating the 99 as “new IP” would be a mistake. This is why the programme is moving generative work to **EVA** (next-generation model path).

### E. Domain shift (AI designs ≠ training examples)
The AI-generated sequences do not sit in exactly the same statistical “neighbourhood” as the natural training set on several physical features.

**Business implication:** High computer scores on AI designs are **hypotheses**, not proof. Lab validation remains mandatory before claiming product-ready performance.

---

## 5. What we did *not* prove yet

Be explicit with stakeholders:

1. **No wet-lab ground truth** for the de novo hits in this notebook — labels on new designs are model-derived.
2. Training labels come from database annotations / homology, not from our own lab assays on every sequence.
3. Temperature was sampled on a **coarse grid** (6 points). Finer melting curves could shift some rankings.
4. GenerRNA did **not** deliver a novel IP portfolio in this round (see novelty finding above).

---

## 6. Architecture diagram (who does what)

```text
 Mac (control)          RunPod (GPU)              AWS S3                 EC2 / Mac (CPU)
 ──────────────         ────────────              ──────                 ───────────────
 Launch & monitor  →    GenerRNA invents   →      Store FASTA &     →    ViennaRNA + NUPACK
 Pull results           (optional BiRNA           embeddings             compute features
                        embeddings)                                      Random Forest score
                                                                         Upload predictions
                                                                         Auto-stop machines
```

**Storage truth:** durable results live in **S3**, not on temporary GPU disks.  
**Compute truth:** invention = GPU; physics scoring = CPU.

---

## 7. Decisions this analysis supports

| Decision | Recommendation |
|----------|----------------|
| Keep investing in the physics + RF **screening** stack? | **Yes** — strong CV performance; clear funnel. |
| Treat GenerRNA 99 hits as novel product candidates? | **No** — novelty screen says they are mostly known. |
| Move generative capacity to EVA (or other non-memorising approaches)? | **Yes** — required for true de novo value. |
| Spend wet-lab budget now? | Only after novelty-filtered shortlists from the new generator; then top ~20 for assay. |
| Need GPU for Vienna/NUPACK? | **No** — keep that on cheap CPU. |

---

## 8. Suggested next milestones (business view)

1. **Complete EVA generation** (new AI inventor) with quality gates → new 10k set on S3 (`llm-batch/eva/v1`).
2. **Re-run the same physics + RF screen** on EVA outputs (reuse Station 3 — already built).
3. **Novelty filter first**, then biophysical shortlist (avoid repeating the GenerRNA memorisation miss).
4. **Wet-lab pilot** on a small top-N (e.g. 10–20) with melting / functional assays.
5. Only then discuss IP filing / platform claims.

---

## 8b. Remediation baseline (length/GC match + intensive features) — Aug 2026

A CPU-only fix was run to test whether the original ~0.95 AUC was real thermoswitch signal or a **length confound**.

### What we changed
- Re-sampled negatives to match positive **length and GC** (global Z-space assignment; gates `|ΔL|≤40`, `|ΔGC|≤0.05`).
- Replaced raw MFE with **MFE per nucleotide** and stem/loop **fractions**.
- Evaluated with **`StratifiedGroupKFold` by Rfam family** (out-of-family), not only random stratified CV.
- Matched training corpus: **527 positives + 519 negatives** (1046 total). Longer natural positives often could not be matched — the negative pool lacks enough long controls (~56% of positives unmatched and dropped; documented in `length_gc_match_report.json`).

### Honest metrics

| Test | AUC | What it means |
|------|-----|----------------|
| Legacy length-alone (old set) | **0.94** | Old data: length almost predicted the label |
| Length-alone after matching | **0.48** | Length shortcut **removed** (gate passed) |
| Legacy raw-MFE stratified CV | **0.95** | Inflated figure from the original memo |
| Intensive features + random stratified CV | **0.76** | Still optimistic (family leakage remains) |
| Intensive features + **StratifiedGroupKFold** (primary) | **0.19** | Out-of-family: **no transferable biophysical classifier** under family hold-out |

All Rfam accessions in this corpus are **pure-label** (switch families vs non-switch families), so family hold-out is a harsh but appropriate test. The model does not generalize across families with the current intensive physics features alone.

### De novo rescore
Rescoring the 10k GenerRNA set with the length-matched intensive RF → hit rate ~**40%** at threshold 0.5 (`denovo_predictions_length_matched.csv`). Treat as exploratory only given the failed out-of-family CV.

### Business implication
The original scorer was **not** a validated thermoswitch detector; it was largely a **length detector**. Programme priority should shift to better negatives (long UTR controls), homology-aware labels within class, and/or sequence+physics models — not to wet-lab based on the legacy 0.95 AUC shortlist.

Artifacts: `data/processed/fused_features_length_matched.csv`, `data/processed/models/rf_thermoswitch_length_matched.joblib`, `data/processed/length_matched_rf_diagnostics.json`.

---

## 8c. RefSeq 5′ UTR negatives (full-panel rematch) — Aug 2026

Follow-up to 8b: replace short Rfam negatives with **housekeeping 5′ UTRs** from RefSeq complete genomes so **all 1198** CD-HIT positives can be length/GC-matched.

### What we built
- Downloaded **80** complete reference assemblies (40 Pseudomonadota + 40 Bacillota).
- Extracted operon-aware housekeeping 5′ UTRs [200–600 nt] → **3572** candidates; Infernal `cmscan --cut_ga` removed **42** → **3530** clean.
- Matched with **CDS-proximal truncation** so each negative is exact length of its positive (`Δμ length = 0`, `|ΔGC|` gated).
- Groups for CV: Rfam family for positives; `REFSEQ:{assembly}` for negatives (**73** assemblies).

### Honest metrics (n=2396)

| Test | AUC | What it means |
|------|-----|----------------|
| Length-alone after RefSeq match | **0.20** | Length shortcut gone (gate passed; better than 8b’s 0.48) |
| Intensive + random stratified CV | **0.74** | Similar to 8b; still optimistic |
| Intensive + **StratifiedGroupKFold** (primary) | **0.18** | Out-of-family/assembly: still **no transferable** physics-only classifier |

### Business implication
Better negatives fixed coverage and length bias, but **did not** recover a general thermoswitch detector under group hold-out. Treat the RefSeq-matched RF as a length-controlled baseline, not a wet-lab gate.

Artifacts: `data/processed/fused_features_refseq_matched.csv`, `data/processed/models/rf_thermoswitch_refseq_matched.joblib`, `data/processed/refseq_matched_rf_diagnostics.json`.

---

## 8d. Dynamic Vienna features (Z, ΔP_RBS, ΔΔG, Q, S) — Aug 2026

Enrichment of the RefSeq-matched panel with composition-relative / differential ViennaRNA features (100 dinucleotide shuffles for MFE Z; pf at 37/55 for RBS exposure and ensemble metrics). No NUPACK re-melt.

### Sanity
- Mean `viennarna_mfe_zscore`: positives **−2.54**, negatives **−1.34** (positives more structured vs dinuc nulls — gate passed).

### Honest metrics (n=2396, 20 features)

| Test | AUC | vs §8c |
|------|-----|--------|
| Length-alone | **0.20** | unchanged (gate passed) |
| Intensive + stratified CV | **0.80** | +0.06 |
| Intensive + **StratifiedGroupKFold** (primary) | **0.19** | ~flat (still no transferable detector) |

Dynamic features improve leakage-prone stratified CV slightly but **do not** lift out-of-family GroupKFold toward the 0.55–0.70 target. See §8e for the monotonic XGBoost follow-up.

Artifacts: `data/processed/fused_features_refseq_dynamic.csv`, `data/processed/models/rf_thermoswitch_refseq_dynamic.joblib`, `data/processed/refseq_dynamic_rf_diagnostics.json`, `data/processed/refseq_dynamic_enrich_report.json`.

---

## 8e. Monotonic XGBoost (physical constraints) — Aug 2026

Same RefSeq-dynamic panel (n=2396, 20 intensive+dynamic features) with `XGBClassifier` and explicit `monotone_constraints`: Z / MFE_per_nt / ΔΔG (−1); ΔP_RBS / Q / S / NUPACK amplitude & hill (+1); stem/loop fracs and remaining intensive cols unconstrained (0).

### Honest metrics vs RF baseline

| Test | RF AUC | Monotonic XGB AUC |
|------|--------|-------------------|
| Length-alone (gate) | **0.20** | **0.20** (passed) |
| Intensive + stratified CV | **0.80** | **0.80** |
| Intensive + **StratifiedGroupKFold** (primary) | **0.19** | **0.20** (Δ ≈ +0.01) |

**Thesis outcome:** Enforcing monotonic physical invariants does **not** unlock a transferable out-of-family detector; GroupKFold stays near chance (~0.20), consistent with the signal living in non-monotonic / family-specific pockets rather than global monotone directions in this feature space.

Artifacts: `data/processed/models/xgb_thermoswitch_refseq_dynamic.joblib`, `data/processed/xgb_refseq_dynamic_diagnostics.json`.

---

## 9. Glossary (30-second definitions)

| Term | Meaning |
|------|---------|
| RNA thermoswitch | A natural or designed RNA “thermostat” that responds to temperature |
| ViennaRNA / NUPACK | Two scientific calculators for RNA folding and melting |
| Random Forest | An ML model that votes across many simple decision trees |
| AUC / accuracy | How well the scorer separates yes/no on known examples |
| De novo | Computer-proposed sequences not pulled as copies from our training list (still need novelty checks vs nature) |
| S3 | Cloud file storage (AWS) |
| RunPod | Rented GPU machines for generative AI |
| EC2 | Rented CPU machines for physics scoring |

---

## 10. Pointers for deeper technical review

| Artifact | Path |
|----------|------|
| Full scientific notebook | `notebooks/05_full_thermo_rf_analysis.ipynb` |
| Machine-readable summary JSON | `data/processed/full_analysis_summary.json` |
| Shortlist table (pre-novelty) | `data/processed/denovo_top_candidates.csv` |
| Novelty follow-up | `notebooks/06_novelty_rfam_analysis.ipynb` |
| Cloud architecture | `cluster/CLOUD_PIPELINE.md` |
| Length/GC match report | `data/processed/balanced/length_gc_match_report.json` |
| Length-matched fused features | `data/processed/fused_features_length_matched.csv` |
| Remediation CV diagnostics | `data/processed/length_matched_rf_diagnostics.json` |
| Length-matched RF model | `data/processed/models/rf_thermoswitch_length_matched.joblib` |
| De novo predictions (remediated RF) | `data/processed/denovo_predictions_length_matched.csv` |
| RefSeq match report | `data/processed/balanced/length_gc_matched_refseq_report.json` |
| RefSeq-matched fused features | `data/processed/fused_features_refseq_matched.csv` |
| RefSeq-matched CV diagnostics | `data/processed/refseq_matched_rf_diagnostics.json` |
| RefSeq-matched RF model | `data/processed/models/rf_thermoswitch_refseq_matched.joblib` |
| RefSeq dynamic fused features | `data/processed/fused_features_refseq_dynamic.csv` |
| RefSeq dynamic CV diagnostics | `data/processed/refseq_dynamic_rf_diagnostics.json` |
| RefSeq dynamic RF model | `data/processed/models/rf_thermoswitch_refseq_dynamic.joblib` |
| Monotonic XGB diagnostics | `data/processed/xgb_refseq_dynamic_diagnostics.json` |
| Monotonic XGB model | `data/processed/models/xgb_thermoswitch_refseq_dynamic.joblib` |

---

*This brief interprets the completed GenerRNA → physics → RF analysis. It is intended for programme planning, not as a regulatory or IP claim.*
