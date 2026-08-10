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
- Stakeholder remediation write-up: `notebooks/05_BUSINESS_BRIEF_thermo_rf_results.md` (§8b–§8d).
- Conda deps for the UTR path: `infernal`, `ncbi-datasets-cli` in `environment.yml`.

### Changed

- Production RF training defaults to the intensive (+ dynamic) feature set instead of raw MFE scalars.
- FASTA header parsing accepts sanitized `REFSEQ:{assembly}` / `assembly:contig` accessions without pipe collisions.
- Fused CSV resume-append aligns to the existing header order to prevent column shift on long thermo batches.

### Fixed

- Length confound that inflated legacy stratified AUC (~0.95) via MFE–length correlation (r ≈ −0.89); length-alone AUC on the rematched panel is ~0.20.

### Results (RefSeq-matched + dynamic features, n=2396)

- Length-alone AUC ≈ **0.20** (length shortcut removed).
- Stratified intensive AUC ≈ **0.80** (optimistic / family leakage).
- StratifiedGroupKFold AUC ≈ **0.19** (no transferable out-of-family thermoswitch detector under unconstrained RF).
- Mean MFE Z: positives ≈ **−2.54** vs RefSeq negatives ≈ **−1.34**.

### Notes

- Large regenerated artifacts (`fused_features_*.csv`, Vienna/NUPACK feature tables, `.joblib` models) remain gitignored under `data/processed/`; rebuild with the commands in `README.md`.
- Follow-up if GroupKFold plateaus near 0.55–0.60 after further modeling: consider XGBoost with monotonic constraints on `viennarna_mfe_zscore` (negative) and `viennarna_delta_P_RBS` (positive).
