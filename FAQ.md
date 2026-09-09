# FAQ

#### 1. Why is GroupKFold AUC ≈ 0.28? Is the model broken?

No. StratifiedGroupKFold grouped on Rfam accession measures **out-of-family** transfer. Leaky stratified CV without groups can look excellent (AUC ≈ 0.97 in historical sidecars) because family-correlated trinucleotides leak across folds. The thesis treats 0.28 as evidence that static 37 °C features do not encode a portable switching rule, which motivates the EVA + yield-gate pivot. See `notebooks/rf_noncircular_methodology.md` and App 04.

#### 2. Do I need ViennaRNA and NUPACK to run `make test`?

No. The 16 pytest cases exercise feature construction, EVA soft filters, and gate helpers without folding engines. Install `uv sync --group biophysics` for ViennaRNA when you rebuild thermo tables or run App 05 sweeps.

#### 3. Why do yield gates ignore NUPACK \(T_m\)?

Empirical Vienna vs NUPACK \(T_m\) parity on the EVA cohort is imperfect. The locked policy is ViennaRNA-only for \(Z\) / \(\Delta P_{\mathrm{RBS}}\) / Hill gates. Dual-engine plots remain useful for concordance audits in App 05.

#### 4. Marimo cells fail with missing CSV / fused panel. What now?

Large tables are gitignored. Either restore them from your local backup or rebuild via [`docs/cookbook.md`](docs/cookbook.md). Apps 03–04 and 06–07 mostly need the committed JSON/CSV sidecars under `data/processed/` and `data/processed/novelty/`.

#### 5. `uv sync` fails or NUPACK import errors?

NUPACK is not on PyPI in this stack. Place the licensed wheel under `nupack-4.1.0.1/package/` and install manually. ViennaRNA comes from the `biophysics` group. Conda users can use `environment.yml` for cd-hit / Infernal / HMMER.

#### 6. How do I point scripts at the repo root on Macleod / SSH?

Run from the clone root. Prefer `PYTHONPATH=src python scripts/...`. Shell wrappers under `scripts/<category>/` set `ROOT` two levels up. Path helpers: `src/data_engineering/paths.py` and `scripts/_repo_paths.py`.

#### 7. Should I open Jupyter `.ipynb` notebooks?

No. Thesis viewers are Marimo `.py` apps (`notebooks/01`–`07`). See [`notebooks/README.md`](notebooks/README.md).

#### 8. Where did Apps 02–04 (prototype / GenerRNA smoke / full RF) go?

Removed from the thesis tree. GenerRNA remains as appendix source under `src/de_novo_hallucinations/genererna/`. Prototype 4-seq fixtures stay under `data/processed/prototype/` for engine smoke tests.

#### 9. Can I use `nohup` for long EVA or thermo jobs?

Prefer `tmux` / `screen` or the detached launcher `scripts/dev/launch_detached.py`. Cluster notes: [`cluster/MACLEOD_EVA.md`](cluster/MACLEOD_EVA.md). Piping through `nohup` often breaks streaming triage similar to the classic minimap2 FAQ warning.

#### 10. What does `.ty_check_output.txt` do?

Nothing. It is a gitignored local scratch file for redirecting `uv check` output. Safe to delete. CI runs `uv check` directly.
