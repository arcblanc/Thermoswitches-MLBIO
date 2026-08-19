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
        Path,
        add_composition_features,
        alt,
        attach_sequences,
        aug_missing_by_class,
        load_balanced_dataset,
        mo,
        np,
        pd,
    )


@app.cell
def _(Path, mo, np, pd):
    def resolve_project_root(start: Path) -> Path:
        """Return the repo root containing ``src/``, walking up from *start*."""
        root = start.resolve()
        if not (root / "src").exists() and (root.parent / "src").exists():
            root = root.parent
        return root

    def cell_status(label: str, ok: bool, detail: str) -> mo.Html:
        """Render a one-line pass/fail badge for a pipeline checkpoint."""
        icon = "✅" if ok else "❌"
        return mo.md(f"{icon} **{label}** — {detail}")

    def load_curation_panel(
        project_root: Path,
        *,
        attach_sequences_fn,
        load_balanced_dataset_fn,
        add_composition_features_fn,
    ) -> tuple[pd.DataFrame, Path, Path, Path]:
        """Load fused physics, attach sequences, merge taxonomy, add k-mers."""
        fused = project_root / "data" / "processed" / "fused_features_refseq_dynamic.csv"
        dataset_csv = (
            project_root
            / "data"
            / "processed"
            / "balanced"
            / "length_gc_matched_refseq_dataset.csv"
        )
        dataset_fasta = (
            project_root
            / "data"
            / "processed"
            / "balanced"
            / "length_gc_matched_refseq_dataset.fasta"
        )
        frame = pd.read_csv(fused)
        frame = attach_sequences_fn(
            frame,
            dataset_csv=str(dataset_csv),
            dataset_fasta=str(dataset_fasta),
        )
        panel = load_balanced_dataset_fn(str(dataset_csv), str(dataset_fasta))
        for col in ("seq_start", "seq_end"):
            frame[col] = frame[col].astype(int)
            panel[col] = panel[col].astype(int)
        tax = panel[
            ["rfamseq_acc", "seq_start", "seq_end", "tax_string", "description", "type"]
        ].drop_duplicates(["rfamseq_acc", "seq_start", "seq_end"])
        frame = frame.merge(tax, on=["rfamseq_acc", "seq_start", "seq_end"], how="left")
        frame = add_composition_features_fn(frame)
        frame["class"] = frame["label"].map(
            {1: "thermoswitch (Rfam)", 0: "RefSeq 5′ UTR"}
        )
        frame["family_or_control"] = np.where(
            frame["label"] == 1, frame["rfam_id"].astype(str), "RefSeq"
        )
        frame["gc_pct"] = frame["viennarna_gc_content"] * 100.0
        return frame, fused, dataset_csv, dataset_fasta

    def class_mean_kmers(
        frame: pd.DataFrame,
        cols: list[str],
        prefix_strip: str,
    ) -> pd.DataFrame:
        """Average k-mer frequencies per class into a long heatmap table."""
        rows: list[dict[str, object]] = []
        for cls, sub in frame.groupby("class"):
            means = sub[cols].mean()
            for col, val in means.items():
                mer = str(col).replace(prefix_strip, "")
                rows.append({"class": cls, "k-mer": mer, "frequency": float(val)})
        return pd.DataFrame(rows)

    def embed_kmer_space(
        frame: pd.DataFrame,
        kmer_cols: list[str],
        *,
        method: str,
        pca_cls,
        scaler_cls,
    ) -> tuple[np.ndarray, str]:
        """Embed scaled k-mer frequencies into 2D coordinates (PCA or UMAP)."""
        matrix = scaler_cls().fit_transform(frame[kmer_cols].fillna(0.0).to_numpy())
        if method == "UMAP":
            try:
                from umap import UMAP

                coords = UMAP(
                    n_components=2,
                    random_state=42,
                    n_neighbors=30,
                    min_dist=0.2,
                ).fit_transform(matrix)
                return coords, "UMAP"
            except Exception as exc:
                coords = pca_cls(n_components=2, random_state=42).fit_transform(matrix)
                return coords, f"PCA (UMAP unavailable: {exc})"
        coords = pca_cls(n_components=2, random_state=42).fit_transform(matrix)
        return coords, "PCA"

    return (
        cell_status,
        class_mean_kmers,
        embed_kmer_space,
        load_curation_panel,
        np,
        resolve_project_root,
    )


@app.cell
def _(
    PROJECT_ROOT,
    add_composition_features,
    attach_sequences,
    cell_status,
    load_balanced_dataset,
    load_curation_panel,
    mo,
    resolve_project_root,
):
    _root = resolve_project_root(PROJECT_ROOT)
    df, FUSED, DATASET_CSV, DATASET_FASTA = load_curation_panel(
        _root,
        attach_sequences_fn=attach_sequences,
        load_balanced_dataset_fn=load_balanced_dataset,
        add_composition_features_fn=add_composition_features,
    )
    load_checks = mo.vstack(
        [
            mo.md("### Pipeline checkpoints"),
            cell_status("project root", _root.exists(), str(_root)),
            cell_status("fused CSV", FUSED.exists(), FUSED.name),
            cell_status("matched CSV", DATASET_CSV.exists(), DATASET_CSV.name),
            cell_status("matched FASTA", DATASET_FASTA.exists(), DATASET_FASTA.name),
            cell_status("panel rows", len(df) == 2396, f"N = {len(df):,}"),
            cell_status(
                "sequences attached",
                df["sequence"].notna().all(),
                f"{int(df['sequence'].notna().sum())} / {len(df)}",
            ),
        ]
    )
    load_checks
    return DATASET_CSV, DATASET_FASTA, FUSED, df, load_checks


@app.cell
def _(aug_missing_by_class, df, mo):
    missing = aug_missing_by_class(df)
    n = int(len(df))
    n_pos = int((df["label"] == 1).sum())
    n_neg = int((df["label"] == 0).sum())
    balance_ok = n_pos == n_neg == 1198
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
    kpi_status = mo.md(
        f"KPI cell: balance **{'OK' if balance_ok else 'FAIL'}** "
        f"({n_pos} pos / {n_neg} neg); missing AUG {missing['n_missing_aug_neg']} neg vs "
        f"{missing['n_missing_aug_pos']} pos."
    )
    return balance_ok, cards, header, kpi_status, missing, n, n_neg, n_pos


@app.cell
def _(cards, header, kpi_status, mo):
    mo.vstack([header, kpi_status, cards])


@app.cell
def _(alt, df, mo):
    pos = df.loc[df["label"] == 1]
    neg = df.loc[df["label"] == 0]
    length_chart = mo.ui.altair_chart(
        alt.Chart(df)
        .transform_density(
            "seq_length",
            as_=["seq_length", "density"],
            groupby=["class"],
            extent=[float(df["seq_length"].min()), float(df["seq_length"].max())],
        )
        .mark_area(opacity=0.45)
        .encode(
            x=alt.X("seq_length:Q", title="Sequence length (nt)"),
            y=alt.Y("density:Q", title="Density"),
            color=alt.Color("class:N", title="Class"),
        )
        .properties(title="Length matching (overlay)", width=420, height=260),
        chart_selection=False,
        legend_selection=False,
    )
    gc_chart = mo.ui.altair_chart(
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
        .properties(title="%GC matching (overlay)", width=420, height=260),
        chart_selection=False,
        legend_selection=False,
    )
    match_md = mo.md(
        rf"""
## Matching diagnostics

Positives: mean L = **{pos['seq_length'].mean():.1f} nt**,
mean %GC = **{pos['gc_pct'].mean():.1f}**.
Negatives: mean L = **{neg['seq_length'].mean():.1f} nt**,
mean %GC = **{neg['gc_pct'].mean():.1f}**.
Overlays should sit on top of each other if matching worked (`|ΔL|≤40`, `|ΔGC|≤0.05`).

Diagnostics cell: length Δ = **{pos['seq_length'].mean() - neg['seq_length'].mean():+.1f} nt**,
%GC Δ = **{pos['gc_pct'].mean() - neg['gc_pct'].mean():+.2f}**.
"""
    )
    mo.vstack([match_md, mo.hstack([length_chart, gc_chart], gap=1)])


@app.cell
def _(df, mo):
    rfam_families = sorted(
        df.loc[df["label"] == 1, "rfam_acc"].dropna().astype(str).unique()
    )
    family_dropdown = mo.ui.dropdown(
        options=rfam_families,
        value=rfam_families[0] if rfam_families else None,
        label="Rfam family (rfam_acc)",
    )
    mo.vstack(
        [
            mo.md(
                f"## Rfam family explorer ({len(rfam_families)} thermoswitch families)"
            ),
            family_dropdown,
        ]
    )
    return family_dropdown, rfam_families


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
    fam_label = (
        fam_df["rfam_id"].dropna().astype(str).iloc[0]
        if fam_df["rfam_id"].dropna().shape[0]
        else str(fam)
    )
    fam_md = mo.md(
        rf"""
**{fam}** — {fam_label} ({len(fam_df)} sequences).
Length mean **{stats.get('mean', float('nan')):.1f} nt**,
std **{stats.get('std', float('nan')):.1f}**,
range **{stats.get('min', float('nan')):.0f}–{stats.get('max', float('nan')):.0f}**.

Family cell: table rows = **{len(table)}**; stem/loop from NUPACK MFE;
`sd_aug_spacing = -1` = no AUG (sentinel).
"""
    )
    mo.vstack([fam_md, mo.ui.table(table, page_size=12)])


@app.cell
def _(DINUC_COLUMNS, TRINUC_COLUMNS, alt, class_mean_kmers, df, mo):
    dinuc_long = class_mean_kmers(df, DINUC_COLUMNS, "dinuc_")
    dinuc_long["b1"] = dinuc_long["k-mer"].str[0]
    dinuc_long["b2"] = dinuc_long["k-mer"].str[1]
    dinuc_hm = mo.ui.altair_chart(
        alt.Chart(dinuc_long)
        .mark_rect()
        .encode(
            x=alt.X("b2:N", title="2nd base"),
            y=alt.Y("b1:N", title="1st base"),
            color=alt.Color("frequency:Q", title="mean freq"),
            facet=alt.Facet("class:N", columns=2),
            tooltip=["k-mer", "class", "frequency"],
        )
        .properties(title="Dinucleotide frequencies (16)", width=220, height=200),
        chart_selection=False,
        legend_selection=False,
    )

    trinuc_long = class_mean_kmers(df, TRINUC_COLUMNS, "trinuc_")
    trinuc_hm = mo.ui.altair_chart(
        alt.Chart(trinuc_long)
        .mark_rect()
        .encode(
            x=alt.X("k-mer:N", sort="-y", axis=alt.Axis(labels=False, ticks=False)),
            y=alt.Y("class:N"),
            color=alt.Color("frequency:Q", title="mean freq"),
            tooltip=["k-mer", "class", "frequency"],
        )
        .properties(title="Trinucleotide frequencies (64)", width=640, height=120),
        chart_selection=False,
        legend_selection=False,
    )
    kmer_status = mo.md(
        f"K-mer cell: dinuc rows = **{len(dinuc_long)}**, trinuc rows = **{len(trinuc_long)}**."
    )
    mo.vstack(
        [
            mo.md("## Dinucleotide / trinucleotide frequency heatmaps"),
            kmer_status,
            dinuc_hm,
            trinuc_hm,
        ]
    )


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
    embed_kmer_space,
    mo,
    reducer,
):
    kmer_cols = [c for c in DINUC_COLUMNS + TRINUC_COLUMNS if c in df.columns]
    coords, method = embed_kmer_space(
        df,
        kmer_cols,
        method=reducer.value,
        pca_cls=PCA,
        scaler_cls=StandardScaler,
    )
    plot_df = df[["class", "family_or_control", "rfam_acc", "seq_length"]].copy()
    plot_df["dim1"] = coords[:, 0]
    plot_df["dim2"] = coords[:, 1]
    color_col = color_by.value
    scatter = mo.ui.altair_chart(
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
        ),
        chart_selection="point",
        legend_selection=True,
    )
    embed_status = mo.md(
        f"Embedding cell: **{method}** on **{len(kmer_cols)}** k-mer features, "
        f"**{plot_df[color_col].nunique()}** {color_col} groups."
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
            embed_status,
            scatter,
        ]
    )


if __name__ == "__main__":
    app.run()
