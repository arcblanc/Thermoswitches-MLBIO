"""Marimo App 1: Rfam & RefSeq dataset curation.

Validates length/%GC matching, family homology, and k-mer sequence space.

Run:
    uv run marimo edit notebooks/marimo_dataset_curation.py
"""

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full", app_title="Rfam & RefSeq Dataset Curation")


@app.cell
def _():
    import sys
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import numpy as np
    import pandas as pd
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    PROJECT_ROOT = Path.cwd().resolve()
    if not (PROJECT_ROOT / "src").exists():
        PROJECT_ROOT = PROJECT_ROOT.parent
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

    from thermo_sim.noncircular_features import (
        DINUC_COLUMNS,
        TRINUC_COLUMNS,
        add_composition_features,
        attach_sequences,
        aug_missing_by_class,
    )
    from thermo_sim.thermo_common import load_balanced_dataset

    alt.data_transformers.disable_max_rows()
    return (
        DINUC_COLUMNS,
        PCA,
        PROJECT_ROOT,
        StandardScaler,
        TRINUC_COLUMNS,
        add_composition_features,
        alt,
        aug_missing_by_class,
        attach_sequences,
        load_balanced_dataset,
        mo,
        np,
        pd,
    )


@app.cell
def _(
    PROJECT_ROOT,
    add_composition_features,
    attach_sequences,
    load_balanced_dataset,
    np,
    pd,
):
    FUSED = PROJECT_ROOT / "data" / "processed" / "fused_features_refseq_dynamic.csv"
    DATASET_CSV = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "balanced"
        / "length_gc_matched_refseq_dataset.csv"
    )
    DATASET_FASTA = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "balanced"
        / "length_gc_matched_refseq_dataset.fasta"
    )

    fused = pd.read_csv(FUSED)
    df = attach_sequences(
        fused,
        dataset_csv=str(DATASET_CSV),
        dataset_fasta=str(DATASET_FASTA),
    )
    panel = load_balanced_dataset(str(DATASET_CSV), str(DATASET_FASTA))
    for _col in ("seq_start", "seq_end"):
        df[_col] = df[_col].astype(int)
        panel[_col] = panel[_col].astype(int)
    tax = panel[
        ["rfamseq_acc", "seq_start", "seq_end", "tax_string", "description", "type"]
    ].drop_duplicates(["rfamseq_acc", "seq_start", "seq_end"])
    df = df.merge(tax, on=["rfamseq_acc", "seq_start", "seq_end"], how="left")
    df = add_composition_features(df)
    df["class"] = df["label"].map({1: "thermoswitch (Rfam)", 0: "RefSeq 5′ UTR"})
    df["family_or_control"] = np.where(
        df["label"] == 1, df["rfam_id"].astype(str), "RefSeq"
    )
    df["gc_pct"] = df["viennarna_gc_content"] * 100.0
    return DATASET_CSV, DATASET_FASTA, FUSED, df


@app.cell
def _(aug_missing_by_class, df, mo):
    missing = aug_missing_by_class(df)
    n = int(len(df))
    n_pos = int((df["label"] == 1).sum())
    n_neg = int((df["label"] == 0).sum())
    cards = mo.hstack(
        [
            mo.stat(
                value=n, label="Total sequences (N)", caption="length/GC-matched panel"
            ),
            mo.stat(
                value=n_pos, label="Positives (Rfam)", caption="known thermoswitches"
            ),
            mo.stat(
                value=n_neg, label="Negatives (RefSeq)", caption="housekeeping 5′ UTRs"
            ),
            mo.stat(
                value=f"{missing['n_missing_aug_neg']} vs {missing['n_missing_aug_pos']}",
                label="Missing AUG (neg vs pos)",
                caption="sentinel — rows not dropped",
            ),
        ],
        justify="space-between",
        gap=1,
    )
    header = mo.md(
        r"""
# App 1 — Rfam & RefSeq dataset curation

Reactive check that the labelled panel is **balanced** and **length/%GC matched**,
then browse Rfam families and k-mer sequence space.

```
Load matched CSV ──► KPI cards (N=2,396, 50/50)
                 ├──► Length & %GC overlay histograms
                 ├──► rfam_acc family explorer
                 └──► Dinuc / trinuc heatmaps + PCA/UMAP
```
"""
    )
    return cards, header, missing, n, n_neg, n_pos


@app.cell
def _(cards, header, mo):
    mo.vstack([header, cards])
    return


@app.cell
def _(alt, df, mo):
    length_chart = (
        alt.Chart(df)
        .transform_density(
            "seq_length",
            as_=["seq_length", "density"],
            groupby=["class"],
            extent=[df["seq_length"].min(), df["seq_length"].max()],
        )
        .mark_area(opacity=0.45)
        .encode(
            x=alt.X("seq_length:Q", title="Sequence length (nt)"),
            y=alt.Y("density:Q", title="Density"),
            color=alt.Color("class:N", title="Class"),
        )
        .properties(title="Length matching (overlay)", width=420, height=260)
    )
    gc_chart = (
        alt.Chart(df)
        .transform_density(
            "gc_pct",
            as_=["gc_pct", "density"],
            groupby=["class"],
            extent=[float(df["gc_pct"].min()), float(df["gc_pct"].max())],
        )
        .mark_area(opacity=0.45)
        .encode(
            x=alt.X("gc_pct:Q", title="%GC"),
            y=alt.Y("density:Q", title="Density"),
            color=alt.Color("class:N", title="Class"),
        )
        .properties(title="%GC matching (overlay)", width=420, height=260)
    )
    match_md = mo.md(
        rf"""
## Matching diagnostics

Positives: mean L = **{df.loc[df["label"] == 1, "seq_length"].mean():.1f} nt**,
mean %GC = **{df.loc[df["label"] == 1, "gc_pct"].mean():.1f}**.
Negatives: mean L = **{df.loc[df["label"] == 0, "seq_length"].mean():.1f} nt**,
mean %GC = **{df.loc[df["label"] == 0, "gc_pct"].mean():.1f}**.
Overlays should sit on top of each other if matching worked (`|ΔL|≤40`, `|ΔGC|≤0.05`).
"""
    )
    mo.vstack([match_md, mo.hstack([length_chart, gc_chart], gap=1)])
    return gc_chart, length_chart, match_md


@app.cell
def _(df, mo):
    families = sorted(df["rfam_acc"].dropna().astype(str).unique())
    family_dropdown = mo.ui.dropdown(
        options=families,
        value=families[0] if families else None,
        label="Rfam family (rfam_acc)",
    )
    mo.vstack(
        [
            mo.md("## Rfam family explorer"),
            family_dropdown,
        ]
    )
    return families, family_dropdown


@app.cell
def _(df, family_dropdown, mo):
    fam = family_dropdown.value
    fam_df = df.loc[df["rfam_acc"].astype(str) == str(fam)].copy()
    show_cols = [
        c
        for c in (
            "rfam_id",
            "rfamseq_acc",
            "seq_length",
            "gc_pct",
            "tax_string",
            "nupack_max_stem_length",
            "nupack_max_loop_length",
            "sd_aug_spacing",
            "sd_aug_missing",
            "sequence",
        )
        if c in fam_df.columns
    ]
    table = fam_df[show_cols].copy()
    if "sequence" in table.columns:
        table["seq_preview"] = table["sequence"].astype(str).str.slice(0, 80)
        table = table.drop(columns=["sequence"])
    stats = fam_df["seq_length"].describe()
    fam_md = mo.md(
        rf"""
**{fam}** — {fam_df["rfam_id"].dropna().astype(str).iloc[0] if fam_df["rfam_id"].dropna().shape[0] else str(fam)}
({len(fam_df)} sequences). Length mean **{stats.get("mean", float("nan")):.1f} nt**,
std **{stats.get("std", float("nan")):.1f}**, range
**{stats.get("min", float("nan")):.0f}–{stats.get("max", float("nan")):.0f}**.

Stem/loop columns are NUPACK MFE motifs; `sd_aug_spacing = -1` means no AUG (sentinel).
"""
    )
    mo.vstack([fam_md, mo.ui.table(table, page_size=12)])
    return fam, fam_df, fam_md, show_cols, stats, table


@app.cell
def _(DINUC_COLUMNS, TRINUC_COLUMNS, alt, df, mo, pd):
    def _class_mean(cols: list[str], prefix_strip: str) -> pd.DataFrame:
        """Average k-mer frequencies per class into a long heatmap table."""
        rows = []
        for cls, sub in df.groupby("class"):
            means = sub[cols].mean()
            for col, val in means.items():
                mer = col.replace(prefix_strip, "")
                rows.append({"class": cls, "k-mer": mer, "frequency": float(val)})
        return pd.DataFrame(rows)

    dinuc_long = _class_mean(DINUC_COLUMNS, "dinuc_")
    dinuc_long["b1"] = dinuc_long["k-mer"].str[0]
    dinuc_long["b2"] = dinuc_long["k-mer"].str[1]
    dinuc_hm = (
        alt.Chart(dinuc_long)
        .mark_rect()
        .encode(
            x=alt.X("b2:N", title="2nd base"),
            y=alt.Y("b1:N", title="1st base"),
            color=alt.Color("frequency:Q", title="mean freq"),
            facet=alt.Facet("class:N", columns=2),
            tooltip=["k-mer", "class", "frequency"],
        )
        .properties(title="Dinucleotide frequencies (16)", width=220, height=200)
    )

    trinuc_long = _class_mean(TRINUC_COLUMNS, "trinuc_")
    trinuc_hm = (
        alt.Chart(trinuc_long)
        .mark_rect()
        .encode(
            x=alt.X("k-mer:N", sort="-y", axis=alt.Axis(labels=False, ticks=False)),
            y=alt.Y("class:N"),
            color=alt.Color("frequency:Q", title="mean freq"),
            tooltip=["k-mer", "class", "frequency"],
        )
        .properties(title="Trinucleotide frequencies (64)", width=640, height=120)
    )
    mo.vstack(
        [
            mo.md("## Dinucleotide / trinucleotide frequency heatmaps"),
            dinuc_hm,
            trinuc_hm,
        ]
    )
    return dinuc_hm, dinuc_long, trinuc_hm, trinuc_long


@app.cell
def _(mo):
    reducer = mo.ui.dropdown(
        options=["PCA", "UMAP"],
        value="PCA",
        label="Embedding",
    )
    color_by = mo.ui.dropdown(
        options=["class", "family_or_control"],
        value="family_or_control",
        label="Color by",
    )
    mo.hstack([reducer, color_by], gap=1)
    return color_by, reducer


@app.cell
def _(
    DINUC_COLUMNS,
    PCA,
    StandardScaler,
    TRINUC_COLUMNS,
    alt,
    color_by,
    df,
    mo,
    reducer,
):
    kmer_cols = [c for c in DINUC_COLUMNS + TRINUC_COLUMNS if c in df.columns]
    X = StandardScaler().fit_transform(df[kmer_cols].fillna(0.0).to_numpy())
    if reducer.value == "UMAP":
        try:
            from umap import UMAP

            coords = UMAP(
                n_components=2,
                random_state=42,
                n_neighbors=30,
                min_dist=0.2,
            ).fit_transform(X)
            method = "UMAP"
        except Exception as exc:
            coords = PCA(n_components=2, random_state=42).fit_transform(X)
            method = f"PCA (UMAP unavailable: {exc})"
    else:
        coords = PCA(n_components=2, random_state=42).fit_transform(X)
        method = "PCA"

    plot_df = df[["class", "family_or_control", "rfam_acc", "seq_length"]].copy()
    plot_df["dim1"] = coords[:, 0]
    plot_df["dim2"] = coords[:, 1]
    color_col = color_by.value
    scatter = (
        alt.Chart(plot_df)
        .mark_circle(size=40, opacity=0.7)
        .encode(
            x=alt.X("dim1:Q", title=f"{method} 1"),
            y=alt.Y("dim2:Q", title=f"{method} 2"),
            color=alt.Color(f"{color_col}:N", title=color_col.replace("_", " ")),
            tooltip=["class", "family_or_control", "rfam_acc", "seq_length"],
        )
        .properties(
            title=f"k-mer sequence space ({method}; 16 dinuc + 64 trinuc)",
            width=720,
            height=420,
        )
        .interactive()
    )
    mo.vstack(
        [
            mo.md(
                r"""
## Sequence-space PCA / UMAP

Points are sequences; axes are a 2D embedding of the **80 k-mer frequencies**.
Color by **class** to check Rfam vs RefSeq overlap, or by **family_or_control**
to see whether Rfam families form homology blobs against a mixed RefSeq cloud.
"""
            ),
            scatter,
        ]
    )
    return X, color_col, coords, kmer_cols, method, plot_df, scatter


if __name__ == "__main__":
    app.run()
