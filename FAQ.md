# FAQ

#### 1. Why is GroupKFold AUC ≈ 0.28?

StratifiedGroupKFold grouped on Rfam accession measures **out-of-family** transfer.
On that split the 92-column Random Forest returned AUC ≈ 0.28.
Random stratified splits without groups can reach AUC ≈ 0.97 when family-correlated trinucleotides leak across folds.
The 0.28 result is the reason the thesis moves to EVA generation and yield filters.
See `notebooks/rf_noncircular_methodology.md` and App 04.

#### 2. Do I need ViennaRNA and NUPACK to run `make test`?

The 16 pytest cases exercise feature construction, EVA soft filters, and gate helpers.
Folding engines are optional for that suite.
Install `uv sync --group biophysics` for ViennaRNA when you rebuild thermo tables or run App 05 sweeps.

#### 3. Why do yield filters use ViennaRNA $T_m$ / $Z$ / $\Delta P_{\mathrm{RBS}}$?

Vienna and NUPACK $T_m$ values can disagree on the same EVA sequences.
The locked policy uses ViennaRNA for $Z$, $\Delta P_{\mathrm{RBS}}$, and Hill filters.
App 05 still plots both engines for concordance audits.

#### 4. Marimo cells fail with missing CSV / fused panel. What now?

Large tables are gitignored.
Restore them from a local backup or rebuild via [`docs/cookbook.md`](docs/cookbook.md).
Apps 03–04 and 06–07 mostly need the committed JSON/CSV sidecars under `data/processed/` and `data/processed/novelty/`.

#### 5. `uv sync` fails or NUPACK import errors?

NUPACK is licensed separately in this stack.
Place the licensed wheel under `nupack-4.1.0.1/package/` and install manually.
ViennaRNA comes from the `biophysics` group.
Conda users can use `environment.yml` for cd-hit / Infernal / HMMER.

#### 6. How do I point scripts at the repo root on Macleod / SSH?

Run from the clone root.
Prefer `PYTHONPATH=src python scripts/...`.
Shell wrappers under `scripts/<category>/` set `ROOT` two levels up.
Path helpers: `src/data_engineering/paths.py` and `scripts/_repo_paths.py`.

#### 7. Which notebook format does the thesis use?

Thesis viewers are Marimo `.py` apps (`notebooks/01`–`07`).
See [`notebooks/README.md`](notebooks/README.md).

#### 8. Where did older prototype / GenerRNA smoke apps go?

They were removed from the thesis app list.
GenerRNA source remains under `src/de_novo_hallucinations/genererna/` as an appendix path.
Prototype 4-seq fixtures stay under `data/processed/prototype/` for engine smoke tests.

#### 9. Can I use `nohup` for long EVA or thermo jobs?

Prefer `tmux` / `screen` or the detached launcher `scripts/dev/launch_detached.py`.
Cluster notes: [`cluster/MACLEOD_EVA.md`](cluster/MACLEOD_EVA.md).

#### 10. What does `.ty_check_output.txt` do?

It is a local typecheck dump used in development.
It is gitignored and safe to delete.
