# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
where version tags are applied.

## [Unreleased]

### Added

- RefSeq housekeeping 5′ UTR negative pipeline: genome download (`scripts/download_refseq_genomes.sh`), UTR extraction (`src/data_engineering/refseq_utr_extract.py`), and Infernal `cmscan --cut_ga` decontamination (`src/data_engineering/cmscan_decontaminate.py`).
- Global length/GC matchmaking (`src/data_engineering/length_gc_match.py`): Z-space cKDTree top-K + Hungarian assignment, hard `|ΔL|≤40` / `|ΔGC|≤0.05` gates, and CDS-proximal truncation for exact-length pairing of all CD-HIT positives.
- Intensive RF feature path in `thermo_classifier.py`: `*_MFE_per_nt`, stem/loop fractions; legacy raw-MFE set retained via `--legacy-features`.
- Vienna dynamic enrichment (`src/thermo_sim/enrich_dynamic_features.py` + helpers in `thermo_common.py` / `vienna_rna.py`): dinucleotide-shuffle MFE Z-score, ΔP_RBS, ΔΔG, ensemble diversity Q, mean positional entropy S.
- Leakage-aware diagnostics (`scripts/rf_length_bias_diagnostics.py`): length-alone gate, stratified contrast, and `StratifiedGroupKFold` by `rfam_acc` / `REFSEQ:{assembly}`.
- Monotonic XGBoost path (`thermo_classifier.py train-xgb` + `scripts/xgb_monotonic_diagnostics.py`): physical `monotone_constraints` on intensive+dynamic features; diagnostic JSON `xgb_refseq_dynamic_diagnostics.json`.
- Non-circular RF inputs (`src/thermo_sim/noncircular_features.py`): static 37 °C physics, 16+64 k-mer frequencies, SD–AUG sentinel (`-1` + `sd_aug_missing`, no row drop), grouped permutation importance.
- Post-hoc module (`src/thermo_sim/rf_posthoc.py` + `thermo_classifier.py posthoc`): OOF confidence bins, ΔP_RBS / Hill / Tm / Z gates, panel-wide Spearman (high-bin \(r_s\) only if \(N \ge 25\)), MW/KS, visual checklist pass rates.
- Cheap \(P_{\mathrm{open,RBS}}\) backfill (`enrich_dynamic_features.py --p-open-only`) without 100-shuffle Z recompute.
- Notebook 07 §§4–7: usable X, non-circular CV, post-hoc, visual diagnostic checklist.
- Results brief: `notebooks/08_noncircular_rf_model_update.md` + `notebooks/08_noncircular_rf_results.ipynb`.
- Hill `bottom`/`top` persisted on new Vienna/NUPACK extracts (no 2396-row melting re-run).

### Changed

- Production RF training defaults to the **non-circular** feature set; `--circular-features` keeps the previous 20-column intensive+dynamic RF; `--legacy-features` keeps raw MFE.
- FASTA header parsing accepts sanitized `REFSEQ:{assembly}` / `assembly:contig` accessions without pipe collisions.
- Fused CSV resume-append aligns to the existing header order to prevent column shift on long thermo batches.

### Fixed

- Length confound that inflated legacy stratified AUC (~0.95) via MFE–length correlation (r ≈ −0.89); length-alone AUC on the rematched panel is ~0.20.

### Results (RefSeq-matched + dynamic features, n=2396)

- Length-alone AUC ≈ **0.20** (length shortcut removed).
- Stratified intensive AUC ≈ **0.80** (optimistic / family leakage).
- StratifiedGroupKFold AUC ≈ **0.19** (unconstrained RF on the circular 20-col set; no transferable out-of-family detector).
- Non-circular RF (static 37 °C + k-mers + SD–AUG): Stratified AUC ≈ **0.97** (family leakage via motifs); StratifiedGroupKFold AUC ≈ **0.28**. High-confidence OOF bin \(N=6\) so high-bin Spearman is underpowered; panel-wide Vienna–NUPACK \(r_s\) is the consensus metric.
- Monotonic XGBoost StratifiedGroupKFold AUC ≈ **0.20** (Δ ≈ +0.01 vs RF) — monotone physical invariants do not rescue out-of-family performance.
- Mean MFE Z: positives ≈ **−2.54** vs RefSeq negatives ≈ **−1.34**.

### Notes

- Large regenerated artifacts (`fused_features_*.csv`, Vienna/NUPACK feature tables, `.joblib` models) remain gitignored under `data/processed/`; rebuild with the commands in `README.md`.
- Monotonic XGBoost GroupKFold plateau near chance (~0.20) suggests remaining signal is family-specific / non-monotonic rather than a global monotone direction in the current feature space.
