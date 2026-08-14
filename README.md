# Thermoswitches-MLBIO

**Author:** Amier Zuhri  
**Domain:** RNA thermoswitches — biophysics, Random Forest ranking, de novo EVA generation

Prokaryotic RNA thermoswitches (RNA thermometers) sit in the 5′ UTR, sequester the Shine-Dalgarno sequence at low temperature, and expose it as temperature rises. This repo builds a labelled physics panel, trains a Random Forest on a **20-feature** matrix for ranking / bootstrapping, then generates new sequences with **EVA** on Macleod GPU and triages them on the Mac.

---

## Final structure

```
Rfam positives  ──┐
                  ├── length/GC match ──► Vienna + NUPACK ──► 20-feature matrix ──► RF
RefSeq 5′ UTRs ──┘                                                      │
                                                                        ▼ ranking aid
GENTEL-Lab EVA 1.4B CLM (frozen) ──► Macleod gpu02 (tmux) ──► generated.fasta
                                                                        │
Mac CPU ──► Vienna Z / ΔP_RBS + Rfam novelty ──► yield candidates
```

| Stage | What it is | Role |
|-------|------------|------|
| **1. Corpus** | Rfam thermoswitch families vs RefSeq housekeeping 5′ UTRs | Honest labelled set (length/GC matched) |
| **2. Features** | 20 intensive + dynamic columns (Vienna + NUPACK) | Physics matrix for the RF |
| **3. RF** | `sklearn` Random Forest (200 trees) | Bootstrap / rank; **not** a wet-lab gate |
| **4. EVA** | Pretrained CLM, TaxID-conditioned `mRNA` | De novo RNA on Macleod MIG |
| **5. Triage** | \(Z\le-2 \land \Delta P_{\mathrm{RBS}}>0 \land E_{\mathrm{Rfam}}>10^{-3}\) | Yield of switch-like, novel sequences |

Code lives under `src/` (`data_engineering/`, `thermo_sim/`, `de_novo_hallucinations/`). Paths resolve from the repo root via `data_engineering.paths.resolve_path()`. Secrets go in `.env`.

---

## 1. Labelled corpus

**Positives — Rfam**  
Prokaryotic heat-shock thermoswitch families → Entrez FASTA → CD-HIT 80% → **1,198** representatives.

**Negatives — NCBI RefSeq**  
Housekeeping 5′ UTRs from complete Pseudomonadota + Bacillota genomes, windows 200–600 nt, Infernal-decontaminated, then **1:1 length/GC matched** to positives (`|ΔL|≤40`, `|ΔGC|≤0.05`, CDS-proximal 3′ end kept). **1,198** matched UTRs.

**Training table:** `data/processed/fused_features_refseq_dynamic.csv` (n = 2,396).  
Groups for out-of-family CV: Rfam accession for positives, `REFSEQ:{assembly}` for negatives.

---

## 2. Twenty-feature matrix

ViennaRNA (melting / unpaired probability) and NUPACK (test-tube ensemble) each contribute intensive scalars. Five Vienna **dynamic** columns are added on top (dinucleotide-shuffle MFE Z, ΔP_RBS on the last 30 nt between 37 °C and 55 °C, ΔΔG, ensemble diversity, mean positional entropy).

| Engine | Features |
|--------|----------|
| NUPACK (8) | `MFE_per_nt`, stem/loop frac, mean exposure, GC, Tm, Hill, amplitude |
| Vienna (7 + 5) | `MFE_per_nt`, loop frac, GC, mean unpaired, Tm, Hill, amplitude, **Z, ΔP_RBS, ΔΔG, Q, S** |

No raw MFE: length is already matched, so the RF sees intensive / composition-relative physics only.

---

## 3. Random Forest (bootstrap)

```bash
python src/thermo_sim/thermo_classifier.py train \
  --fused-csv data/processed/fused_features_refseq_dynamic.csv \
  --model-path data/processed/models/rf_thermoswitch_refseq_dynamic.joblib
```

| Check | Result |
|-------|--------|
| Length-alone AUC | ~0.20 (length shortcut removed) |
| Stratified CV | ~0.80 (family leakage — optimistic) |
| **StratifiedGroupKFold** | **~0.19** (no transferable out-of-family detector) |

The RF is a **ranking aid** on the 20-feature matrix. Yield of generated RNA is decided by the locked biophysical + novelty gates below, not by RF probability.

---

## 4. EVA on Macleod

EVA is **not trained here**. GENTEL-Lab’s **1.4B CLM** is hosted on `gpu02` (A100 MIG `3g.20gb`). This repo’s `eva_generate.py` chunks the official `eva-generate` CLI (`--rna_type mRNA --taxid 562`), persists raw FASTA, **soft-drops** low-complexity sequences, and tops up to quota.

tmux lives on **macleod1** (not gpu02) so an SSH drop does not kill the GPU job. Architecture sketch: [`cluster/EVA_OVERVIEW.md`](cluster/EVA_OVERVIEW.md). Operator install: [`cluster/MACLEOD_EVA.md`](cluster/MACLEOD_EVA.md).

**Completed E. coli run (T = 0.9, chunk 128):**

| Panel | Accepted | Yield (three gates) |
|-------|----------|---------------------|
| Pilot | 512 | 31 / 512 = **6.05%** |
| Top-up | 1,488 | 74 / 1,488 = **4.97%** |
| **All** | **2,000** | **105 / 2,000 = 5.25%** |

Yield gate: \(Z\le-2 \land \Delta P_{\mathrm{RBS}}>0 \land E_{\mathrm{Rfam}}>10^{-3}\) (no Rfam hit ⇒ \(E=\infty\)). Stream novelty no-hit stayed ~50%. Candidates: `data/processed/eva_pilot/top_candidates.fasta` and `data/processed/eva_stream/top_candidates.fasta`.

---

## Setup

### Mac — environment

```bash
git clone https://github.com/arcblanc/Thermoswitches-MLBIO.git
cd Thermoswitches-MLBIO

conda env create -f environment.yml
conda activate thermoswitches-mlbio
# or: python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
#     (then install cd-hit, viennarna, infernal, blast, hmmer separately)

cp .env.example .env   # set EMAIL and NCBI_API_KEY
```

Install the NUPACK 4.1 wheel from `nupack-4.1.0.1/package/` (paid license; gitignored).

### Mac — corpus + 20 features + RF

```bash
# Positives
python src/data_engineering/data_extraction.py
python src/data_engineering/sequence_retrieval.py --all
python src/data_engineering/cd_hit_sequence_similarity.py --all

# Negatives (script sources .env; aborts if NCBI_API_KEY is missing)
bash scripts/download_refseq_genomes.sh
PYTHONPATH=src python src/data_engineering/refseq_utr_extract.py
PYTHONPATH=src python src/data_engineering/cmscan_decontaminate.py
PYTHONPATH=src python src/data_engineering/length_gc_match.py \
  --all-positives --cds-truncate \
  --negatives-csv data/processed/refseq_utr/candidates_clean.csv \
  --negatives-fasta data/processed/refseq_utr/candidates_clean.fasta

# Thermo fold + dynamic enrich → fused_features_refseq_dynamic.csv
python src/thermo_sim/thermo_batch.py --run
PYTHONPATH=src python src/thermo_sim/enrich_dynamic_features.py --workers 4

# RF
python src/thermo_sim/thermo_classifier.py train \
  --fused-csv data/processed/fused_features_refseq_dynamic.csv \
  --model-path data/processed/models/rf_thermoswitch_refseq_dynamic.joblib
```

### Macleod — EVA generate

From the Mac (F5 tunnel): `ssh -p 1024 t41am25@127.0.0.1`.

```bash
# Login node — tmux first, then GPU
module load tmux
tmux new -s eva
srun -p gpu -w gpu02 --gres=gpu:3g.20gb:1 --pty bash -i

module load python/3.11.9 cuda/12.4.0 gcc/12.3.0 git
source ~/sharedscratch/venvs/torch_mod/bin/activate
export CUDA_VISIBLE_DEVICES=0 CUDA_MODULE_LOADING=LAZY PYTHONPATH=src

# One-shot install + 4-seq smoke (clone ~/EVA, HF checkpoint, eva-generate)
bash scripts/macleod_eva_install_smoke.sh
```

Generate (inside the same tmux / `srun`; detach with Ctrl-b d):

```bash
cd ~/Thermoswitches-MLBIO
export STORAGE_TARGET=local
export EVA_RNA_TYPE=mRNA
export EVA_CHECKPOINT_DIR=~/eva_checkpoint/EVA_1.4B_CLM
export EVA_SMOKE_SINGLE_HOST=1
export EVA_NUM_SAMPLES=2000
export EVA_CHUNK_SIZE=128
export EVA_OUT=data/processed/de_novo/generated.fasta

python src/de_novo_hallucinations/eva_generate.py --smoke --resume \
  --chunk-size 128 \
  --output-fasta "$EVA_OUT" \
  --skip-s3-probe \
  --temperature 0.9
```

Attach later from **macleod1** only: `module load tmux && tmux attach -t eva`. Do not attach tmux from inside gpu02.

### Mac — triage generated FASTA

```bash
# While GPU runs (poll every 300 s, 250-seq slices, seed past the 512 pilot)
PYTHONPATH=src python scripts/eva_stream_triage.py \
  --source ssh --remote t41am25@127.0.0.1 --ssh-port 1024 \
  --remote-fasta /home/t41am25/Thermoswitches-MLBIO/data/processed/de_novo/generated.fasta \
  --stride 250 --seed-count 512 --poll-seconds 300 --workers 4

# After 2000/2000, leftover <250:
PYTHONPATH=src python scripts/eva_stream_triage.py \
  --source ssh --remote t41am25@127.0.0.1 --ssh-port 1024 \
  --remote-fasta /home/t41am25/Thermoswitches-MLBIO/data/processed/de_novo/generated.fasta \
  --stride 250 --seed-count 512 --workers 4 \
  --once --flush-remainder
```

When generation is done, `exit` the `srun` to free the MIG, then `tmux kill-session -t eva` on macleod1.

---

## Artifacts

| Path | Contents |
|------|----------|
| `data/processed/fused_features_refseq_dynamic.csv` | 20-feature labelled panel |
| `data/processed/models/rf_thermoswitch_refseq_dynamic.joblib` | RF |
| `data/processed/de_novo/generated.fasta` | EVA accepted sequences (cluster) |
| `data/processed/eva_pilot/top_candidates.fasta` | 31 pilot passers |
| `data/processed/eva_stream/top_candidates.fasta` | 74 stream passers |
| `cluster/MACLEOD_EVA.md` | Install, smoke, generate |
| `cluster/EVA_OVERVIEW.md` | Why tmux / orchestrator / triage |
| `cluster/macleod_log.md` | Session log |
