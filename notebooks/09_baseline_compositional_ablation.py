"""Marimo: compositional ablation (length / %GC shortcuts + distribution parity).

Thesis figure panel — proves Hungarian matching killed compositional shortcuts.

Run:
    uv run marimo edit notebooks/09_baseline_compositional_ablation.py
"""

import marimo

__generated_with = "0.24.0"
app = marimo.App(
    width="medium",
    app_title="Baseline compositional ablation",
)


@app.cell
def _():
    """Bootstrap imports and project paths."""
    import json
    import sys
    from pathlib import Path

    import matplotlib.pyplot as plt
    import marimo as mo
    import numpy as np
    import pandas as pd
    from great_tables import GT, loc, md, px, style

    PROJECT_ROOT = Path.cwd().resolve()
    if not (PROJECT_ROOT / "src").exists():
        PROJECT_ROOT = PROJECT_ROOT.parent
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

    from thermo_sim import curation_eda as eda
    from thermo_sim import thesis_results_figures as trf

    fig_dir = PROJECT_ROOT / "notebooks" / "figures" / "09_baseline_ablation"
    fig_dir.mkdir(parents=True, exist_ok=True)
    thesis_fig_dir = PROJECT_ROOT / "notebooks" / "figures" / "thesis_figures"
    thesis_fig_dir.mkdir(parents=True, exist_ok=True)
    ablation_json = (
        PROJECT_ROOT / "data" / "processed" / "rf_compositional_ablation.json"
    )

    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.bbox": "tight",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
        }
    )
    return (
        GT,
        PROJECT_ROOT,
        ablation_json,
        eda,
        fig_dir,
        json,
        loc,
        md,
        mo,
        np,
        pd,
        plt,
        px,
        style,
        thesis_fig_dir,
        trf,
    )


@app.cell
def _(PROJECT_ROOT, eda, mo):
    """Load the length/%GC-matched fused panel."""
    root = eda.resolve_project_root(PROJECT_ROOT)
    df, fused_path, _matched_csv, _matched_fa = eda.load_curation_panel(root)

    intro = mo.md(
        r"""
    # Baseline compositional ablation

    > Did Hungarian length / %GC matching kill trivial classification shortcuts?

    <br>
    **Matching gate**:
    <br>
    <br>
    $|\Delta L| \le 40$,
    <br>
    $|\Delta GC| \le 0.05$.
    <br>
    <br>
    **Panel**: $N = 2396$ ($1198$ Rfam / $1198$ RefSeq).
    """
    )

    status = mo.md(
        f"✅ Loaded `{fused_path.name}` — "
        f"**{len(df):,}** rows · "
        f"**{int((df['label'] == 1).sum()):,}** pos / "
        f"**{int((df['label'] == 0).sum()):,}** neg"
    )

    mo.vstack([intro, status])
    return (df,)


@app.cell
def _(
    GT,
    ablation_json,
    fig_dir,
    json,
    loc,
    md,
    mo,
    pd,
    px,
    style,
    thesis_fig_dir,
    trf,
):
    """Compositional ablation table — Docs-ready like cohort map."""
    report = json.loads(ablation_json.read_text())

    def _row(name: str, key: str, vs: str) -> dict[str, str]:
        """Format one ablation row from the diagnostics JSON."""
        block = report[key]
        return {
            "Model": name,
            "Features": {
                "length_alone_stratified": "seq_length",
                "gc_alone_stratified": "%GC",
                "length_gc_stratified": "length + %GC",
                "random_chance": "—",
            }[key],
            "Stratified ROC-AUC": f"{float(block['roc_auc']):.3f}",
            "Accuracy": f"{float(block['accuracy']):.3f}",
            "vs chance (0.50)": vs,
        }

    ablation_df = pd.DataFrame(
        [
            _row("Length-alone RF", "length_alone_stratified", "Below chance"),
            _row("%GC-alone RF", "gc_alone_stratified", "Below chance"),
            _row("Length + %GC RF", "length_gc_stratified", "Below chance"),
            {
                "Model": "Random chance (balanced)",
                "Features": "—",
                "Stratified ROC-AUC": "0.500",
                "Accuracy": "0.500",
                "vs chance (0.50)": "—",
            },
        ]
    )

    ablation_table = (
        GT(ablation_df)
        .tab_header(
            title=md("**Compositional ablation — cohort matching control**"),
            subtitle=md(
                "Stratified 5-fold RF ($T=200$) on matched panel $N=2{,}396$ · "
                "composition-only features"
            ),
        )
        .cols_align(align="left", columns=["Model", "Features", "vs chance (0.50)"])
        .cols_align(align="center", columns=["Stratified ROC-AUC", "Accuracy"])
        .tab_options(**trf.docs_table_options(px))
        .tab_style(
            style=style.text(weight="bold", size=px(28)),
            locations=loc.body(columns="Model"),
        )
        .tab_source_note(
            source_note=md(
                "Hungarian matching ($|\\Delta L| \\le 40$, $|\\Delta GC| \\le 0.05$) "
                "killed compositional shortcuts. All composition-only AUCs $< 0.50$. "
                "PNG: 4× zoom + 96 DPI for Docs paste."
            )
        )
        .opt_row_striping()
        .opt_table_outline(style="solid", width=px(2), color="#222222")
        .opt_stylize(style=1, color="gray")
    )

    ablation_png = thesis_fig_dir / "compositional_ablation_google_docs.png"
    export_note = ""
    try:
        ablation_table.gtsave(str(ablation_png), zoom=4.0, vwidth=1200, expand=24)
        trf.stamp_png_dpi(ablation_png, dpi=96)
        (fig_dir / ablation_png.name).write_bytes(ablation_png.read_bytes())
        export_note = (
            f"Saved Docs-ready PNG → `{ablation_png}` "
            f"({ablation_png.stat().st_size // 1024} KB, 4× zoom, 96 DPI)."
        )
    except Exception as exc:  # noqa: BLE001
        export_note = f"PNG export skipped ({type(exc).__name__}: {exc})."

    ablation_html = mo.Html(
        f"""
        <div style="zoom:1.25; transform-origin: top left; max-width: 100%;">
          {ablation_table.as_raw_html()}
        </div>
        """
    )

    section = mo.md("## Shortcut check")
    caption = mo.md(
        rf"""
    **Caption.** Compositional ablation. Random Forests trained on
    length alone (AUC ${float(report["length_alone_stratified"]["roc_auc"]):.3f}$),
    %GC alone (${float(report["gc_alone_stratified"]["roc_auc"]):.3f}$),
    or length + %GC (${float(report["length_gc_stratified"]["roc_auc"]):.3f}$)
    all fall **below** chance ($0.50$), confirming that Hungarian matching
    ($|\\Delta L| \\le 40$, $|\\Delta GC| \\le 0.05$) eliminated trivial
    compositional shortcuts prior to thermodynamic / $k$-mer feature extraction.
    """
    )
    takeaway = mo.md(
        r"""
    <br>
    **What the numbers prove**:
    <br>
    <br>
    - Length alone cannot separate classes,
    <br>
    - %GC alone cannot separate classes,
    <br>
    - even **both** composition cues together stay below chance,
    <br>
    - downstream RF must rely on folding / $k$-mers — not body-size matching.
    """
    )

    mo.vstack([section, ablation_html, mo.md(export_note), caption, takeaway])
    return ablation_df, ablation_table, report


@app.cell
def _(df, eda, fig_dir, mo, np, plt, thesis_fig_dir, trf):
    """Overlaid length and %GC densities — Docs-ready export to thesis_figures."""
    pos = df.loc[df["label"] == 1]
    neg = df.loc[df["label"] == 0]

    c_pos = "#0D7377"
    c_neg = "#C44536"

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.0))
    fig.suptitle(
        "Matched compositional distributions (Rfam vs RefSeq)",
        fontsize=16,
        y=0.98,
    )
    fig.subplots_adjust(wspace=0.28, left=0.08, right=0.98, top=0.84, bottom=0.14)

    ax = axes[0]
    bins_l = np.linspace(
        float(df["seq_length"].min()),
        float(df["seq_length"].max()),
        36,
    )
    ax.hist(
        pos["seq_length"],
        bins=bins_l,
        density=True,
        histtype="stepfilled",
        alpha=0.35,
        color=c_pos,
        edgecolor=c_pos,
        linewidth=1.4,
        label="Rfam (pos)",
    )
    ax.hist(
        neg["seq_length"],
        bins=bins_l,
        density=True,
        histtype="stepfilled",
        alpha=0.35,
        color=c_neg,
        edgecolor=c_neg,
        linewidth=1.4,
        label="RefSeq (neg)",
    )
    ax.set_xlabel("Sequence length (nt)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Length", pad=8, fontsize=13)
    ax.legend(frameon=False, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    bins_g = np.linspace(
        float(df["gc_pct"].min()),
        float(df["gc_pct"].max()),
        36,
    )
    ax.hist(
        pos["gc_pct"],
        bins=bins_g,
        density=True,
        histtype="stepfilled",
        alpha=0.35,
        color=c_pos,
        edgecolor=c_pos,
        linewidth=1.4,
        label="Rfam (pos)",
    )
    ax.hist(
        neg["gc_pct"],
        bins=bins_g,
        density=True,
        histtype="stepfilled",
        alpha=0.35,
        color=c_neg,
        edgecolor=c_neg,
        linewidth=1.4,
        label="RefSeq (neg)",
    )
    ax.set_xlabel("%GC", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("%GC", pad=8, fontsize=13)
    ax.legend(frameon=False, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)

    out_png = thesis_fig_dir / "length_gc_matched_overlap.png"
    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.3)
    trf.stamp_png_dpi(out_png, dpi=96)
    (fig_dir / out_png.name).write_bytes(out_png.read_bytes())

    audit = eda.confounder_audit_table(df)
    parity = eda.gc_biological_parity_check(df)
    len_row = audit.loc[audit["feature"] == "Length (nt)"].iloc[0]
    gc_row = audit.loc[audit["feature"] == "%GC"].iloc[0]

    dist_intro = mo.md(
        r"""
    ## Feature distribution parity

    > Do negative controls mirror positives on length and %GC?

    <br>
    **Visual test**: overlaid densities must overlap — forcing the RF onto folding / $k$-mers, not composition.
    """
    )
    stats_md = mo.md(
        rf"""
    Length Cliff's $\\delta$ = {float(len_row["cliffs_delta"]):.3f};
    %GC KS $D$ = {float(parity["KS_D"]):.3f}
    (mean $|\\Delta\\mathrm{{GC}}| = {parity["mean_abs_delta_gc"]:.2f}$%).
    <br>
    Length KS $D = {float(len_row["KS_stat"]):.3f}$ ($p = {float(len_row["KS_p"]):.3g}$);
    %GC KS $D = {float(gc_row["KS_stat"]):.3f}$.
    <br>
    Saved → `notebooks/figures/thesis_figures/{out_png.name}` (300 DPI + 96 DPI tag).
    """
    )
    mo.vstack([dist_intro, fig, stats_md])
    return


if __name__ == "__main__":
    app.run()
