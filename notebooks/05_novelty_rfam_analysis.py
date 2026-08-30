import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Rfam Novelty Analysis — BLAST + nhmmer

    Evaluates whether the 99 RF-filtered de novo thermoswitch candidates are:

    1. **Identical / near-identical** (≥90% identity to Rfam) — likely memorization
    2. **Remote homologs** (<90% identity, E≤0.1) — novel but evolutionarily related
    3. **No hits** (E>0.1 from both tools) — de novo hallucination

    Methods mirror GenerRNA (relaxed nhmmer E-value 0.1) with Rfam 14.9 as reference.

    Run the pipeline first:
    ```bash
    bash scripts/extraction/download_rfam.sh
    bash scripts/extraction/run_novelty_search.sh
    ```
    """)
    return


@app.cell
def _():
    # '%matplotlib inline' command supported automatically in marimo

    import json
    import sys
    from pathlib import Path

    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    PROJECT_ROOT = Path.cwd().resolve()
    if not (PROJECT_ROOT / "src").exists():
        PROJECT_ROOT = PROJECT_ROOT.parent
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

    NOVELTY_DIR = PROJECT_ROOT / "data" / "processed" / "novelty"
    FIG_DIR = NOVELTY_DIR / "figures"
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid", context="notebook", palette="colorblind")
    return FIG_DIR, NOVELTY_DIR, PROJECT_ROOT, json, pd, plt


@app.cell
def _(NOVELTY_DIR, PROJECT_ROOT, json, pd):
    report = pd.read_csv(NOVELTY_DIR / "novelty_report.csv")
    summary = json.loads((NOVELTY_DIR / "novelty_summary.json").read_text())
    candidates = pd.read_csv(
        PROJECT_ROOT / "data" / "processed" / "denovo_top_candidates.csv"
    )

    merged = report.merge(
        candidates[["record_id", "prob_positive", "priority_score"]],
        on="record_id",
        how="left",
    )
    print(
        f"Candidates: {len(report)} | E-value threshold: {summary['evalue_threshold']}"
    )
    pd.DataFrame(summary["categories"]).T
    return merged, report, summary


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Bucket distribution
    """)
    return


@app.cell
def _(FIG_DIR, plt, report):
    order = ["identical_near", "remote_homolog", "no_hit"]
    labels = {
        "identical_near": "Identical / near-identical (≥90%)",
        "remote_homolog": "Remote homolog (<90%)",
        "no_hit": "No hit (E>0.1)",
    }
    counts = report["novelty_category"].value_counts().reindex(order, fill_value=0)
    _fig, ax = plt.subplots(figsize=(7, 4))
    counts.plot.bar(ax=ax, color=["#DD8452", "#4C72B0", "#55A868"])
    ax.set_xticklabels(
        [labels.get(k, k) for k in counts.index], rotation=15, ha="right"
    )
    ax.set_ylabel("count")
    ax.set_title("Novelty categories — 99 de novo candidates vs Rfam 14.9")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 1, f"{v} ({v / len(report):.0%})", ha="center")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "novelty_buckets.png")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Identity distribution by tool
    """)
    return


@app.cell
def _(FIG_DIR, plt, report):
    _fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    blast_ids = report["blast_identity_pct"].dropna()
    nhmmer_ids = report["nhmmer_identity_pct"].dropna()
    axes[0].hist(blast_ids, bins=20, color="#4C72B0", edgecolor="white")
    axes[0].axvline(90, color="red", ls="--", lw=1)
    axes[0].set_xlabel("BLAST % identity (best hit)")
    axes[0].set_title(f"blastn hits: {blast_ids.notna().sum()}")
    axes[1].hist(nhmmer_ids, bins=20, color="#DD8452", edgecolor="white")
    axes[1].axvline(90, color="red", ls="--", lw=1)
    axes[1].set_xlabel("nhmmer % identity (best hit)")
    axes[1].set_title(f"nhmmer hits: {nhmmer_ids.notna().sum()}")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "identity_histograms.png")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Top remote homologs (thesis table)
    """)
    return


@app.cell
def _(merged):
    remote = merged[merged["novelty_category"] == "remote_homolog"].sort_values(
        "best_identity_pct", ascending=False
    )
    cols = [
        "record_id",
        "prob_positive",
        "best_tool",
        "best_target_id",
        "best_identity_pct",
        "best_evalue",
        "best_alignment_length",
    ]
    remote[cols].head(20)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Methods paragraph (template)

    Edit counts after pipeline run.
    """)
    return


@app.cell
def _(summary):
    c = summary["categories"]
    methods = f"""
    Novelty of 99 physics-filtered de novo candidates was assessed against Rfam 14.9
    (Rfam.fa) using blastn (-task blastn, E≤{summary["evalue_threshold"]}) and nhmmer
    (--rna, -E {summary["evalue_threshold"]}). Per-sequence identity was defined as matching
    nucleotides divided by alignment length (GenerRNA convention). The best hit across both
    tools was retained. Sequences with ≥90% identity were classified as near-identical
    (n={c["identical_near"]["count"]}), <90% as remote homologs (n={c["remote_homolog"]["count"]}),
    and sequences with no hit at E≤{summary["evalue_threshold"]} as no-hit (n={c["no_hit"]["count"]}).
    """.strip()
    print(methods)
    return


if __name__ == "__main__":
    app.run()
