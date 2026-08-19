# EVA on Macleod — architecture + tmux run (overview)

EVA is **not trained in this repo**. We use GENTEL-Lab’s pretrained **1.4B CLM** checkpoint and wrap it: generate on `gpu02`, triage on the Mac.

Operator install/smoke: [`MACLEOD_EVA.md`](MACLEOD_EVA.md). Session log: [`macleod_log.md`](macleod_log.md).

---

## Why this stack

| Piece | Reason |
|-------|--------|
| EVA 1.4B CLM | Sequence LM that can emit RNA given TaxID + `mRNA` |
| MoE / flash-attn | Official inference path; needs ~16 GB VRAM → MIG `3g.20gb` |
| Our orchestrator | Chunk + soft-drop so 3 bad seqs don’t kill a 6 h batch |
| tmux on **macleod1** | `srun` dies if SSH drops; tmux on the login node keeps the job |
| Mac CPU triage | Vienna / blastn / nhmmer don’t need the GPU |

---

## Architecture (what we “built”)

```
GENTEL-Lab EVA (pretrained)
        │
        ├── Tokenizer + CLM (causal LM)
        ├── Optional MoE experts (needs flash-attn)
        └── Conditioning: --rna_type mRNA  --taxid 562 (E. coli)
                │
                ▼
This repo: eva_generate.py  (orchestrator, not the weights)
        │
        ├── chunk 128 seqs → eva-generate CLI
        ├── persist RAW fasta first
        ├── soft-drop (format / length / 3-mer collapse)
        └── append passers → generated.fasta  until N=2000
                │
                ▼
Mac: eva_stream_triage.py
        ├── Vienna: Z, ΔP_RBS
        ├── blastn + nhmmer vs Rfam
        └── yield: Z≤−2 ∧ ΔP_RBS>0 ∧ E_Rfam>1e−3
```

---

## Process tree (tmux session)

```
Mac
 └── F5 tunnel  →  ssh -p 1024  macleod1
                      │
                      ├── tmux session  eva2k     ← lives HERE (login node)
                      │      └── srun gpu02 MIG
                      │             └── module python/cuda
                      │             └── venv torch_mod
                      │             └── eva_generate.py --resume
                      │                    └── eva-generate (GPU)
                      │
                      └── (optional) second SSH to peek;
                          do NOT tmux attach from inside gpu02
```

**Why tmux on macleod1, not gpu02?**  
The GPU allocation is a child of `srun`. tmux sessions on `gpu02` disappear when that job ends. Session `eva2k` on the login node outlives your laptop.

**Why not Mac tmux for generation?**  
The model never runs on the Mac. Mac only rsyncs FASTA and scores it.

---

## Run ladder (reasons)

```
1. srun + CUDA smoke          prove MIG + glibc/Python
2. eva-generate 4 seqs        prove checkpoint
3. orchestrator --smoke       prove soft-drop + paths
4. 512 E. coli (chunk 128)    yield ≥0.5–2%?  → Path A
5. 2000 --resume              top-up, keep the 512
6. Mac stream triage          250-seq slices while GPU runs
7. --once --flush-remainder   last <250 seqs after 2000/2000
```

---

## Mental model

```
Weights (frozen)     →  GPU writes RNA
Orchestrator         →  don’t lose a chunk to one bad seq
tmux                 →  don’t lose the GPU job to F5
Triage (Mac)         →  “is this switch-like and novel?”
```
