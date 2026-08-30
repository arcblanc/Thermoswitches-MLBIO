"""Marimo App 1: Rfam vs RefSeq curation & exploratory EDA.

Five reactive cells — cohort KPIs, length/%GC matching audit, 37 °C biophysics,
SD–AUG geometry, and k-mer manifold — with plain-language analogies.

Run:
    uv run marimo edit notebooks/01_rfam_refseq_curation_eda.py
"""

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full", app_title="App 1 · Rfam vs RefSeq Curation EDA")


@app.cell
def _():
    """Bootstrap imports, Altair, and typed curation helpers."""
    import sys
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import pandas as pd

    PROJECT_ROOT = Path.cwd().resolve()
    if not (PROJECT_ROOT / "src").exists():
        PROJECT_ROOT = PROJECT_ROOT.parent
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

    from thermo_sim import curation_eda as eda

    alt.data_transformers.disable_max_rows()
    return PROJECT_ROOT, alt, eda, mo, pd


@app.cell
def _(PROJECT_ROOT, eda, mo):
    """Load the matched panel once and show pipeline checkpoints."""
    root = eda.resolve_project_root(PROJECT_ROOT)
    df_all, fused_path, csv_path, fasta_path = eda.load_curation_panel(root)

    def _badge(label: str, ok: bool, detail: str) -> mo.Html:
        """One-line pass/fail checkpoint badge."""
        icon = "✅" if ok else "❌"
        return mo.md(f"{icon} **{label}** — {detail}")

    intro = mo.md(
        r"""
    # App 1 · Rfam vs RefSeq curation & exploratory EDA

    **Question in plain words:** After we matched thermoswitches to ordinary 5′ UTRs
    on length and %GC, do the two groups still look different — in folding energy,
    ribosome-binding geometry, or sequence “dialect”?

    **Analogy:** Think of matching as pairing runners of the same height and weight
    before comparing race times. If times still differ, the difference is not just
    body size — something else is going on. Here length and %GC are the “body size”
    we already balanced.
    """
    )
    checkpoints = mo.vstack(
        [
            mo.md("### Pipeline checkpoints"),
            _badge("project root", root.exists(), str(root)),
            _badge("fused CSV", fused_path.exists(), fused_path.name),
            _badge("matched CSV", csv_path.exists(), csv_path.name),
            _badge("matched FASTA", fasta_path.exists(), fasta_path.name),
            _badge("panel rows", len(df_all) == 2396, f"N = {len(df_all):,}"),
            _badge(
                "class balance",
                int((df_all["label"] == 1).sum()) == int((df_all["label"] == 0).sum()),
                f"{int((df_all['label'] == 1).sum())} pos / "
                f"{int((df_all['label'] == 0).sum())} neg",
            ),
        ]
    )
    return checkpoints, df_all, intro


@app.cell
def _(df_all, eda, mo):
    """Reactive controls: class filter + Rfam family dropdown."""
    class_toggle = mo.ui.dropdown(
        options=["Both classes", "Rfam positives", "RefSeq negatives"],
        value="Both classes",
        label="Class filter",
    )
    family_dd = mo.ui.dropdown(
        options=eda.family_choices(df_all),
        value="All families",
        label="Rfam family (rfam_acc)",
    )
    controls = mo.hstack([class_toggle, family_dd], justify="start", gap=1.0)
    return class_toggle, controls, family_dd


@app.cell
def _(class_toggle, df_all, eda, family_dd):
    """Apply reactive filters to the working frame."""
    df = eda.filter_panel(
        df_all,
        class_filter=str(class_toggle.value),
        family=str(family_dd.value),
    )
    return (df,)


@app.cell
def _(controls, df, eda, intro, mo, pd):
    """KPI cards: balance, missing AUG, families, missingness flags."""
    kpi = eda.kpi_summary(df)
    miss_df = pd.DataFrame(
        [{"feature": k, "frac_missing": v} for k, v in kpi["missingness"].items()]
    )

    cards = mo.hstack(
        [
            mo.stat(
                label="Samples in view",
                value=f"{kpi['n']:,}",
                caption="after class / family filters",
            ),
            mo.stat(
                label="Positives (Rfam)",
                value=f"{kpi['n_pos']:,}",
                caption="label = 1",
            ),
            mo.stat(
                label="Negatives (RefSeq)",
                value=f"{kpi['n_neg']:,}",
                caption="label = 0",
            ),
            mo.stat(
                label="Unique Rfam families",
                value=str(kpi["n_families"]),
                caption="among positives in view",
            ),
            mo.stat(
                label="Missing AUG (Rfam)",
                value=str(kpi["n_missing_aug_pos"]),
                caption="sd_aug_missing = 1",
            ),
            mo.stat(
                label="Missing AUG (RefSeq)",
                value=str(kpi["n_missing_aug_neg"]),
                caption="sd_aug_missing = 1",
            ),
        ],
        justify="space-between",
        gap=0.5,
        wrap=True,
    )

    cell1 = mo.vstack(
        [
            intro,
            mo.md("## Cell 1 · Cohort ingestion & KPI cards"),
            mo.md(
                """
    **What to look for:** Roughly equal positives and negatives, and how often the
    start codon (AUG) is missing from the window we annotated.

    **Analogy:** A missing AUG is like a book with the first page torn out — we can
    still read the middle, but we mark those shelves with a bright sticker
    (`sd_aug_spacing = −1`) so they do not get treated as a real spacer length.
    """
            ),
            controls,
            cards,
            mo.md("### Missingness flags (fraction NaN)"),
            mo.ui.table(miss_df, selection=None),
            mo.md(
                "**Balance flag:** "
                + (
                    "✅ balanced in view."
                    if kpi["balanced"]
                    else (
                        "⚠️ unbalanced in view "
                        "(expected when a single class or family is selected)."
                    )
                )
            ),
        ]
    )
    cell1
    return


@app.cell
def _(alt, df, eda, mo):
    """Length/%GC parity: step densities, eCDFs, Q–Q, KS + Wasserstein."""
    audit = eda.confounder_audit_table(df)
    color = alt.Color("class:N", title="Class")

    def density_chart(column: str, title: str, x_title: str) -> alt.Chart:
        """Paired step-density (area) histograms for one confounder."""
        return (
            alt.Chart(df)
            .transform_density(column, as_=[column, "density"], groupby=["class"])
            .mark_area(opacity=0.35, interpolate="step")
            .encode(
                x=alt.X(f"{column}:Q", title=x_title),
                y=alt.Y("density:Q", title="Density"),
                color=color,
                tooltip=["class:N", f"{column}:Q", "density:Q"],
            )
            .properties(width=320, height=220, title=title)
        )

    def ecdf_chart(column: str, title: str, x_title: str) -> alt.Chart:
        """Empirical CDF — “how much of the class sits below this value?”."""
        ecdf = eda.ecdf_frame(df, column)
        return (
            alt.Chart(ecdf)
            .mark_line(interpolate="step-after")
            .encode(
                x=alt.X(f"{column}:Q", title=x_title),
                y=alt.Y("ecdf:Q", title="Fraction ≤ x"),
                color=color,
                tooltip=["class:N", f"{column}:Q", "ecdf:Q"],
            )
            .properties(width=320, height=220, title=title)
        )

    def qq_chart(column: str, title: str) -> alt.Chart:
        """Q–Q scatter; points on y = x mean identical quantile shapes."""
        qq = eda.qq_frame(df, column)
        if qq.empty:
            return (
                alt.Chart(qq)
                .mark_point()
                .properties(title=title, width=280, height=220)
            )
        line = (
            alt.Chart(qq)
            .mark_line(color="gray", strokeDash=[4, 4])
            .encode(x="q_neg:Q", y="q_neg:Q")
        )
        pts = (
            alt.Chart(qq)
            .mark_circle(size=40, opacity=0.7)
            .encode(
                x=alt.X("q_neg:Q", title="RefSeq quantile"),
                y=alt.Y("q_pos:Q", title="Rfam quantile"),
                tooltip=["q_neg:Q", "q_pos:Q"],
            )
        )
        return (line + pts).properties(width=280, height=220, title=title)

    length_panel = density_chart(
        "seq_length", "Length density", "Length (nt)"
    ) | ecdf_chart("seq_length", "Length eCDF", "Length (nt)")
    gc_panel = density_chart("gc_pct", "%GC density", "%GC") | ecdf_chart(
        "gc_pct", "%GC eCDF", "%GC"
    )
    qq_panel = qq_chart("seq_length", "Q–Q length") | qq_chart("gc_pct", "Q–Q %GC")

    gc_parity = eda.gc_biological_parity_check(df)
    _d = gc_parity["KS_D"]
    _delta = gc_parity["mean_abs_delta_gc"]
    _d_txt = "n/a" if _d is None else f"{float(_d):.4f}"
    _delta_txt = f"{float(_delta):.3f}"
    _p = gc_parity["KS_p"]
    _p_txt = "n/a" if _p is None else f"{float(_p):.4g}"
    _mean_pos = f"{float(gc_parity['mean_gc_pos']):.2f}"
    _mean_neg = f"{float(gc_parity['mean_gc_neg']):.2f}"
    if gc_parity["passed"]:
        _parity_icon = "✅"
        _parity_verdict = (
            "biologically minor — residual %GC shift sits inside the 5% matching gate"
        )
    else:
        _parity_icon = "⚠️"
        _parity_verdict = (
            "not minor by the $D < 0.06$ and "
            r"$\overline{\Delta GC} \le 1.5\%$ thresholds — inspect matching"
        )
    # Keep the display equation on one line: a leading "-" after a newline
    # inside $$…$$ is parsed as a Markdown list and drops the minus.
    _gc_eq = (
        r"$$\overline{\Delta GC} = \bigl\lvert "
        r"\mathrm{mean}(\%GC_{\mathrm{pos}}) - "
        r"\mathrm{mean}(\%GC_{\mathrm{neg}}) \bigr\rvert$$"
    )
    gc_parity_md = mo.md(
        f"""
    ### Biological-minor %GC check

    {_gc_eq}

    | Metric | Value | Gate |
    |--------|-------|------|
    | KS $D$ (%GC) | `{_d_txt}` | $D < 0.06$ → `{gc_parity["d_ok"]}` |
    | mean $\\lvert\\Delta GC\\rvert$ | `{_delta_txt}` % | $\\le 1.5\\%$ → `{gc_parity["delta_ok"]}` |
    | mean %GC (pos / neg) | `{_mean_pos}` / `{_mean_neg}` | — |
    | KS $p$ (%GC) | `{_p_txt}` | informative only |

    {_parity_icon} **Verdict:** {_parity_verdict}.

    **Why this check:** with $N \\approx 2400$, KS $p$ can reject tiny shape differences that
    do not open a useful classification shortcut. Prefer $D$ and mean $|\\Delta GC|$
    alongside $p > 0.05$.
    """
    )

    cell2 = mo.vstack(
        [
            mo.md("## Cell 2 · Matching audit (length & %GC)"),
            mo.md(
                r"""
    **Visual choice:**
    <br>
    *step densities* show the shape of each distribution;
    <br>
    *eCDFs* make small shifts obvious;
    <br>
    *Q–Q plots* ask “do the ordered values line up on the diagonal?”;
    <br>
    the **KS test** asks if the two clouds could still be the same population ($p > 0.05$ is what we want after matching).

    **Analogy:** KS is a strict twin test — if two silhouette cutouts still
    overlap almost perfectly, matching did its job and classifiers cannot cheat
    by memorizing “long GC-rich = switch.”

    From what I can see, it looks like, Rfseq has a larger spread density in length around 300 nt - 500 nt what seems to be double the hill of RFAM postive thermo switches
    <br>
    length ecdf seems to only show rfam thermoswitches. maybe its completely overlapped so there is not much shifts.
    <br>
    %GC shows similar refseq doubling of density similar to length
    <br>
    gc ecdf shows slight shifts from 30 to 40 %gc

    the Q-Q of length and GC both are straight lines. Though qq length has a gap between 200 and 300 quantile.
    """
            ),
            mo.ui.altair_chart(length_panel),
            mo.ui.altair_chart(gc_panel),
            mo.ui.altair_chart(qq_panel),
            mo.md("### Confounder statistics (want KS $p > 0.05$)"),
            mo.ui.table(audit, selection=None),
            gc_parity_md,
        ]
    )
    cell2
    return


@app.cell
def _(df, eda, mo):
    """Split violins for MFE/nt, Q, S, stem/loop fractions + MW / Cliff's delta."""
    long = eda.biophysics_long(df)
    contrast = eda.biophysics_contrast_table(df)
    fig_violin = eda.plot_biophysics_violins(long)

    cell3 = mo.vstack(
        [
            mo.md("## Cell 3 · Static 37 °C biophysical landscape"),
            mo.md(
                r"""
    **Visual choice:** Faceted **violin + inner box** plots for intensive ground-state
    metrics ($\mathrm{MFE}/N$, ensemble diversity $Q$, positional entropy $S$,
    stem/loop fractions). The violin shows the full shape of each class; the box
    inside marks the median and IQR.

    **Analogy:** MFE/nt is how tightly the RNA is "zipped" per nucleotide at rest.
    Ensemble diversity $Q$ is how many different outfits the molecule can wear
    in the Boltzmann closet — high $Q$ means a flexible wardrobe, not one locked
    pose. We ask: do natural thermoswitches sit in a different thermodynamic
    room than matched non-switches?
    """
            ),
            fig_violin,
            mo.md(
                "### Mann–Whitney $U$ + Cliff’s $\\delta$ "
                "(effect size: $|\\delta|<0.147$ negligible, "
                "$<0.33$ small, $<0.474$ medium)"
            ),
            mo.ui.table(contrast, selection=None),
        ]
    )
    cell3
    return


@app.cell
def _(alt, df, eda, mo):
    """SD–AUG spacing histogram + P_paired,RBS(37) vs %GC scatter + χ²."""
    spacing = df.copy()
    spacing["aug_status"] = spacing["sd_aug_spacing"].apply(
        lambda x: "missing AUG (−1)" if int(x) < 0 else "AUG present"
    )
    present = spacing.loc[spacing["sd_aug_spacing"] >= 0]

    hist = (
        alt.Chart(present)
        .mark_bar(opacity=0.7)
        .encode(
            x=alt.X(
                "sd_aug_spacing:Q", bin=alt.Bin(step=1), title="SD→AUG spacing (nt)"
            ),
            y=alt.Y("count()", title="Count"),
            color=alt.Color("class:N", title="Class"),
            tooltip=["class:N", "count()"],
        )
        .properties(width=420, height=240, title="SD–AUG spacing (AUG present only)")
    )
    sentinel = (
        alt.Chart(spacing.loc[spacing["sd_aug_spacing"] < 0])
        .mark_bar(opacity=0.85, color="#b35c1e")
        .encode(
            x=alt.X("class:N", title="Class"),
            y=alt.Y("count()", title="Missing-AUG count"),
            tooltip=["class:N", "count()"],
        )
        .properties(width=260, height=240, title="Missing AUG sentinel (−1)")
    )

    _scatter_rbs = (
        alt.Chart(df.dropna(subset=["P_paired_RBS_37", "gc_pct"]))
        .mark_circle(opacity=0.45, size=35)
        .encode(
            x=alt.X("gc_pct:Q", title="%GC"),
            y=alt.Y("P_paired_RBS_37:Q", title="P_paired,RBS (37 °C)"),
            color=alt.Color("class:N", title="Class"),
            tooltip=["class:N", "gc_pct:Q", "P_paired_RBS_37:Q", "rfam_id:N"],
        )
        .properties(
            width=480,
            height=280,
            title="Baseline RBS pairing vs composition",
        )
    )

    chi = eda.sd_aug_chi_square(df)
    spacing_tbl = eda.spacing_summary(df)
    chi_md = (
        f"χ² = {chi['chi2']:.3f}, dof = {chi['dof']}, p = {chi['p']:.3g}"
        if chi["chi2"] is not None
        else "χ² not computable for current filter (need both classes & both AUG states)."
    )

    cell4 = mo.vstack(
        [
            mo.md("## Cell 4 · RBS & translation-initiation geometry"),
            mo.md(
                r"""
    **Visual choice:** A **1-nt bin histogram** for SD→AUG distance (canonical
    window ≈ 5–10 nt) plus a separate **sentinel bar chart** for missing AUGs so
    the −1 flag never pollutes the real spacer axis. The **scatter** of
    $P_{\mathrm{paired,RBS}}(37^\circ\mathrm{C})$ vs %GC asks whether RBS occlusion
    is just a GC story.

    **Analogy:** The Shine–Dalgarno motif is a “docking bumper” a few nucleotides
    upstream of the start codon. Spacing is parking distance — too close or too
    far and the ribosome struggles to park. The −1 sticker means “no start codon
    in this annotated window,” not “zero nucleotides.”
    """
            ),
            mo.ui.altair_chart(hist | sentinel),
            mo.ui.altair_chart(_scatter_rbs),
            mo.md("### Spacing summary + missing-AUG χ²"),
            mo.ui.table(spacing_tbl, selection=None),
            mo.md(f"**χ² (class × missing AUG):** {chi_md}"),
        ]
    )
    cell4
    return


@app.cell
def _(alt, df, eda, mo):
    """PCA/UMAP on 16+64 k-mers, family sizing, BH-FDR enrichment table."""
    method = mo.ui.dropdown(options=["UMAP", "PCA"], value="UMAP", label="Embedding")
    color_by = mo.ui.dropdown(
        options=["class", "family_or_control"],
        value="class",
        label="Color points by",
    )
    kmer_cols = [
        c for c in (*eda.DINUC_COLUMNS, *eda.TRINUC_COLUMNS) if c in df.columns
    ]

    coords, used = eda.embed_kmer_space(df, kmer_cols, method=str(method.value))
    emb = df[["class", "family_or_control", "rfam_id", "label"]].copy()
    emb["dim1"] = coords[:, 0]
    emb["dim2"] = coords[:, 1]
    # Size by family abundance among positives (controls get median size).
    fam_counts = (
        emb.loc[emb["label"] == 1, "family_or_control"].value_counts().to_dict()
    )
    emb["family_n"] = emb["family_or_control"].map(fam_counts).fillna(10).astype(float)

    _scatter_kmer = (
        alt.Chart(emb)
        .mark_circle(opacity=0.65)
        .encode(
            x=alt.X("dim1:Q", title=f"{used} 1"),
            y=alt.Y("dim2:Q", title=f"{used} 2"),
            color=alt.Color(f"{color_by.value}:N", title=str(color_by.value)),
            size=alt.Size(
                "family_n:Q",
                title="Family n (pos)",
                scale=alt.Scale(range=[20, 200]),
            ),
            tooltip=[
                "class:N",
                "family_or_control:N",
                "rfam_id:N",
                "family_n:Q",
                "dim1:Q",
                "dim2:Q",
            ],
        )
        .properties(
            width=520,
            height=400,
            title=f"k-mer manifold ({used}) — 16 dinuc + 64 trinuc",
        )
        .interactive()
    )

    dinuc_heat = eda.class_mean_kmers(
        df, [c for c in eda.DINUC_COLUMNS if c in df.columns], "dinuc_"
    )
    heat = (
        alt.Chart(dinuc_heat)
        .mark_rect()
        .encode(
            x=alt.X("k-mer:N", title="Dinucleotide"),
            y=alt.Y("class:N", title=None),
            color=alt.Color(
                "frequency:Q", title="Mean freq", scale=alt.Scale(scheme="viridis")
            ),
            tooltip=["class:N", "k-mer:N", "frequency:Q"],
        )
        .properties(width=520, height=90, title="Mean dinucleotide dialect by class")
    )

    enrich = eda.kmer_enrichment_table(df, kmer_cols)
    top = enrich.head(25) if not enrich.empty else enrich

    cell5 = mo.vstack(
        [
            mo.md("## Cell 5 · k-mer sequence manifold & clade diversity"),
            mo.md(
                r"""https://lichess.org/
    **Visual choice:** **UMAP/PCA** compresses 80 k-mer frequencies into a 2D map
    so islands of similar “dialect” become visible. Point **size** tracks how
    common each Rfam family is (large blobs = abundant clades). The dinucleotide
    **heatmap** is the same idea without dimensionality reduction. BH-FDR
    $t$-tests list which specific $k$-mers differ after multiple-testing control.

    **Analogy:** Each sequence is a smoothie recipe with 80 ingredient ratios.
    UMAP places similar recipes near each other on a café table map. If Rfam
    points form isolated islands, a tree model can memorize the island address
    instead of learning a true thermoswitch rule — that is the “clade
    memorization” risk this cell documents.
    """
            ),
            mo.hstack([method, color_by], justify="start", gap=1.0),
            mo.ui.altair_chart(_scatter_kmer),
            mo.ui.altair_chart(heat),
            mo.md("### Top k-mer contrasts (Welch $t$, Benjamini–Hochberg FDR)"),
            mo.ui.table(top, selection=None),
            mo.md(
                f"Significant at FDR ≤ 0.05: "
                f"**{int(enrich['significant'].sum()) if not enrich.empty else 0}** / "
                f"{len(enrich)} tested features."
            ),
        ]
    )
    cell5
    return


@app.cell
def _(checkpoints, mo):
    """Footer: checkpoints + how to re-run."""
    mo.vstack(
        [
            checkpoints,
            mo.md(
                """
    ---
    **Reproduce**

    ```bash
    uv run marimo edit notebooks/01_rfam_refseq_curation_eda.py
    ```

    Helpers live in `src/thermo_sim/curation_eda.py` (typed, docstringed, Ruff-clean).
    """
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
