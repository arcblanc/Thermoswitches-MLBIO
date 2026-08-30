"""Marimo app: thesis figures for Thermoswitches-MLBIO.

Publication-ready tables and panels for the Methods / Results narrative.

Run:
    uv run marimo edit notebooks/thesis_figures.py
"""

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full", app_title="Thesis Figures")


@app.cell
def _():
    """Bootstrap imports for thesis figure cells."""
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

    from thermo_sim import thesis_results_figures as trf

    fig_dir = PROJECT_ROOT / "notebooks" / "figures" / "thesis_figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    data_dir = PROJECT_ROOT / "data" / "processed"
    return (
        GT,
        PROJECT_ROOT,
        data_dir,
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
        sys,
        trf,
    )


@app.cell
def _(GT, fig_dir, loc, md, mo, pd, px, style, trf):
    """Figure 1 — cohort map (Docs-sized fonts + high-DPI PNG)."""
    # Fonts sized for Google Docs page-width paste (~6.5").
    # Prior 18px body in a ~1600px-wide table shrank to ~5 pt on the page.
    cohort_df = pd.DataFrame(
        [
            {
                "Cohort": "Positive RNATs",
                "Source": "Rfam",
                "Size (N)": "1,198",
                "Target Label": "Positive (1)",
                "Feature Representation": "Static Tabular Features",
                "Role in Workflow": "Supervised RF Training",
            },
            {
                "Cohort": "Negative Controls",
                "Source": "NCBI RefSeq",
                "Size (N)": "1,198",
                "Target Label": "Negative (0)",
                "Feature Representation": "Static Tabular Features",
                "Role in Workflow": "Supervised RF Training",
            },
        ]
    )

    cohort_table = (
        GT(cohort_df)
        .tab_header(
            title=md("**Cohort map for the thermoswitch workflow**"),
            subtitle=md("Labelled Rfam / RefSeq panel for supervised RF training"),
        )
        .cols_align(
            align="left",
            columns=[
                "Cohort",
                "Source",
                "Feature Representation",
                "Role in Workflow",
            ],
        )
        .cols_align(align="center", columns=["Size (N)", "Target Label"])
        .tab_options(**trf.docs_table_options(px))
        .tab_style(
            style=style.text(weight="bold", size=px(28)),
            locations=loc.body(columns="Cohort"),
        )
        .tab_source_note(
            source_note=md(
                "Matched Rfam positives and RefSeq negatives (*N* = 1,198 each) "
                "used for supervised RF training. PNG: 4× zoom + 96 DPI for Docs paste."
            )
        )
        .opt_row_striping()
        .opt_table_outline(style="solid", width=px(2), color="#222222")
        .opt_stylize(style=1, color="gray")
    )

    cohort_png = fig_dir / "cohort_map_google_docs.png"
    _cohort_export_note = ""
    try:
        # zoom=4 → sharp pixels; vwidth matches fixed table so fonts stay large.
        cohort_table.gtsave(
            str(cohort_png),
            zoom=4.0,
            vwidth=1200,
            expand=24,
        )
        trf.stamp_png_dpi(cohort_png, dpi=96)
        _cohort_export_note = (
            f"Saved Docs-ready PNG → `{cohort_png}` "
            f"({cohort_png.stat().st_size // 1024} KB, 4× zoom, 96 DPI)."
        )
    except Exception as exc:  # noqa: BLE001 — Chrome may be missing in some envs
        _cohort_export_note = (
            f"PNG export skipped ({type(exc).__name__}: {exc}). "
            "Table still renders in-notebook at enlarged type."
        )

    # CSS zoom keeps on-screen text large when screenshotting into Docs.
    cohort_html = mo.Html(
        f"""
        <div style="zoom:1.25; transform-origin: top left; max-width: 100%;">
          {cohort_table.as_raw_html()}
        </div>
        """
    )

    intro = mo.md(
        r"""
    # Thesis figures

    Publication panels for the thermoswitch Methods / Results narrative.

    > Which labelled cohorts enter supervised RF training?

    <br>
    **What this table shows**:
    <br>
    <br>
    - Rfam positives (*N* = 1,198) as label 1,
    <br>
    - RefSeq negatives (*N* = 1,198) as label 0.
    <br>
    <br>
    **Google Docs tip**: import `cohort_map_google_docs.png` (large type + 4× zoom + 96 DPI).
    If it still looks short, drag a corner to page width — text should stay readable.
    """
    )

    mo.vstack([intro, cohort_html, mo.md(_cohort_export_note)])
    return


@app.cell
def _(GT, PROJECT_ROOT, data_dir, fig_dir, loc, md, mo, np, pd, plt, px, style, trf):
    """Cohort matching + control baseline — Docs-ready table + density overlap."""
    report = trf.load_json(data_dir / "rf_compositional_ablation.json")

    def _ablation_row(name: str, key: str, vs: str) -> dict[str, str]:
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
            _ablation_row("Length-alone RF", "length_alone_stratified", "Below chance"),
            _ablation_row("%GC-alone RF", "gc_alone_stratified", "Below chance"),
            _ablation_row("Length + %GC RF", "length_gc_stratified", "Below chance"),
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

    ablation_png = fig_dir / "compositional_ablation_google_docs.png"
    ablation_note = ""
    try:
        ablation_table.gtsave(str(ablation_png), zoom=4.0, vwidth=1200, expand=24)
        trf.stamp_png_dpi(ablation_png, dpi=96)
        ablation_note = (
            f"Saved Docs-ready table → `{ablation_png.name}` "
            f"({ablation_png.stat().st_size // 1024} KB, 4× zoom, 96 DPI)."
        )
    except Exception as exc:  # noqa: BLE001
        ablation_note = f"Ablation PNG export skipped ({type(exc).__name__}: {exc})."

    # Density overlap (length / %GC) — same Docs-ready DPI as cohort map.
    fused = pd.read_csv(
        PROJECT_ROOT / "data" / "processed" / "fused_features_refseq_dynamic.csv",
        usecols=["label", "seq_length", "viennarna_gc_content"],
    )
    fused["gc_pct"] = fused["viennarna_gc_content"].astype(float) * 100.0
    pos = fused.loc[fused["label"] == 1]
    neg = fused.loc[fused["label"] == 0]
    c_pos, c_neg = "#0D7377", "#C44536"
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.0))
    fig.suptitle(
        "Matched compositional distributions (Rfam vs RefSeq)",
        fontsize=16,
        y=0.98,
    )
    fig.subplots_adjust(wspace=0.28, left=0.08, right=0.98, top=0.84, bottom=0.14)

    ax = axes[0]
    bins_l = np.linspace(
        float(fused["seq_length"].min()), float(fused["seq_length"].max()), 36
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
    bins_g = np.linspace(float(fused["gc_pct"].min()), float(fused["gc_pct"].max()), 36)
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

    overlap_png = fig_dir / "length_gc_matched_overlap.png"
    fig.savefig(overlap_png, dpi=300, bbox_inches="tight", pad_inches=0.3)
    trf.stamp_png_dpi(overlap_png, dpi=96)
    plt.close(fig)

    ablation_html = mo.Html(
        f"""
        <div style="zoom:1.25; transform-origin: top left; max-width: 100%;">
          {ablation_table.as_raw_html()}
        </div>
        """
    )
    section = mo.md(
        rf"""
    ## Cohort matching and control baseline

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
    **What the numbers prove**:
    <br>
    <br>
    - Length alone AUC ${float(report["length_alone_stratified"]["roc_auc"]):.3f}$ (below chance),
    <br>
    - %GC alone AUC ${float(report["gc_alone_stratified"]["roc_auc"]):.3f}$,
    <br>
    - length + %GC AUC ${float(report["length_gc_stratified"]["roc_auc"]):.3f}$.
    <br>
    <br>
    Saved → `{ablation_png.name}`, `{overlap_png.name}` (4× / 300 DPI + 96 DPI tag).
    """
    )
    mo.vstack([section, ablation_html, mo.md(ablation_note), fig])
    return


@app.cell
def _(GT, fig_dir, loc, md, mo, px, style, trf):
    """Methods — Table 2a/2b software stack (Docs import-safe split PNGs)."""
    software_a_df, software_b_df = trf.build_thesis_software_tables()

    def _software_gt(frame, title: str, subtitle: str):
        return (
            GT(frame)
            .tab_header(title=md(title), subtitle=md(subtitle))
            .cols_align(
                align="left",
                columns=["Category", "Package / Tool", "Primary Function in Pipeline"],
            )
            .cols_align(align="center", columns=["Version / Source"])
            .tab_options(**trf.docs_table_options(px))
            .tab_style(
                style=style.text(weight="bold", size=px(24)),
                locations=loc.body(columns="Category"),
            )
            .tab_style(
                style=style.text(weight="bold", size=px(24)),
                locations=loc.body(columns="Package / Tool"),
            )
            .tab_source_note(
                source_note=md(
                    "Methods software inventory. Import both PNGs into Docs if the "
                    "single-table file exceeds upload limits. "
                    "4592 px width · 96 DPI · PNG compressed."
                )
            )
            .opt_row_striping()
            .opt_table_outline(style="solid", width=px(2), color="#222222")
            .opt_stylize(style=1, color="gray")
        )

    software_a_table = _software_gt(
        software_a_df,
        "**Table 2a: Software stack — runtime, ML, bioinformatics**",
        "Python · scikit-learn · SciPy · CD-HIT · Infernal · BLAST+ · HMMER",
    )
    software_b_table = _software_gt(
        software_b_df,
        "**Table 2b: Software stack — biophysics and generative model**",
        "ViennaRNA · NUPACK · EVA 1.4B CLM · PyTorch / CUDA",
    )

    software_a_png = fig_dir / "software_stack_a_google_docs.png"
    software_b_png = fig_dir / "software_stack_b_google_docs.png"
    _software_export_note = ""
    try:
        trf.export_docs_gt_png(
            software_a_table,
            software_a_png,
            zoom=trf.DOCS_TABLE_TALL_ZOOM,
            target_width=trf.DOCS_TABLE_TALL_TARGET_WIDTH,
        )
        trf.export_docs_gt_png(
            software_b_table,
            software_b_png,
            zoom=trf.DOCS_TABLE_TALL_ZOOM,
            target_width=trf.DOCS_TABLE_TALL_TARGET_WIDTH,
        )
        _software_export_note = (
            f"Saved Docs-ready PNGs → `{software_a_png.name}` "
            f"({software_a_png.stat().st_size // 1024} KB), "
            f"`{software_b_png.name}` ({software_b_png.stat().st_size // 1024} KB) · "
            f"width={trf.DOCS_TABLE_TALL_TARGET_WIDTH}, 96 DPI, 3× zoom."
        )
    except Exception as exc:  # noqa: BLE001 — Chrome may be missing in some envs
        _software_export_note = (
            f"PNG export skipped ({type(exc).__name__}: {exc}). "
            "Tables still render in-notebook."
        )

    software_a_html = mo.Html(
        f"""
        <div style="zoom:1.25; transform-origin: top left; max-width: 100%;">
          {software_a_table.as_raw_html()}
        </div>
        """
    )
    software_b_html = mo.Html(
        f"""
        <div style="zoom:1.25; transform-origin: top left; max-width: 100%;">
          {software_b_table.as_raw_html()}
        </div>
        """
    )

    software_intro = mo.md(
        r"""
    ## Software and compute stack

    > Which packages and CLI tools appear in the Methods software inventory?

    <br>
    **Table 2a** — runtime, ML, bioinformatics ($N=9$).
    <br>
    **Table 2b** — biophysics + generative model ($N=4$).
    <br>
    <br>
    **Docs tip**: import `software_stack_a_google_docs.png` then `software_stack_b_google_docs.png`
    separately if a single tall PNG fails to upload.
    """
    )

    mo.vstack(
        [
            software_intro,
            software_a_html,
            software_b_html,
            mo.md(_software_export_note),
        ]
    )
    return


@app.cell
def _(fig_dir, mo, trf):
    """Figure 2 — square layout: EVA bar on top; RF | biophysics below."""
    # Flat subgraphs (no double-nest) so G→ThermalSweep attaches inside that box.
    # Same boxes/colours; De Novo + Rfam Positive enter Thermal Sweep.
    workflow_mermaid_src = r"""
---
config:
  flowchart:
    htmlLabels: false
    padding: 14
    nodeSpacing: 20
    rankSpacing: 32
    wrappingWidth: 120
    curve: linear
---
flowchart TB
    classDef data fill:#f9f9f9,stroke:#333,stroke-width:1px;
    classDef process fill:#e1f5fe,stroke:#0288d1,stroke-width:2px;
    classDef model fill:#e8f5e9,stroke:#388e3c,stroke-width:2px;
    classDef output fill:#fff3e0,stroke:#f57c00,stroke-width:2px;
    classDef denovo fill:#fffde7,stroke:#f9a825,stroke-width:3px;

    subgraph evaBubble ["EVA generation"]
        direction LR
        D["`EVA Foundation
Model`"]:::model
        E["`Latent Space
Sampling`"]:::process
        F["`Yield-Gated
Filtering`"]:::process
        G["`De Novo Candidates
N=105`"]:::denovo
        D --> E --> F --> G
    end

    subgraph rfCol ["Supervised RF"]
        direction TB
        A2["`RefSeq Negative
N=1198`"]:::data
        A1["`Rfam Positive
N=1198`"]:::data
        B["`Feature Extraction
K-mers + Static Thermo Features`"]:::process
        C["`Random Forest Classifier
5-fold grouped CV`"]:::model
        A2 ~~~ A1
        A2 --> B
        A1 --> B
        B --> C
    end

    subgraph bioCol ["In silico biophysics"]
        direction TB
        H["`Biophysical
Pipeline`"]:::process
        ThermalSweep["`Thermal Sweep
37C to 42C`"]:::process
        J["`Hill Metric
Extraction`"]:::output
        H --> ThermalSweep --> J
    end

    %% Keep RF | bio side-by-side (same ranks)
    A2 ~~~ H
    A1 ~~~ ThermalSweep
    B ~~~ ThermalSweep
    C ~~~ J

    %% Attach De Novo and Rfam Positive into Thermal Sweep box
    G --> ThermalSweep
    A1 --> ThermalSweep

    style evaBubble fill:#e8f5e9,stroke:#388e3c,stroke-width:2px,color:#1b5e20
    style rfCol fill:#fafafa,stroke:#616161,stroke-width:1px
    style bioCol fill:#fff8e1,stroke:#f57c00,stroke-width:1px
"""

    workflow_diagram = mo.mermaid(workflow_mermaid_src)

    workflow_intro = mo.md(
        r"""
    ## Workflow topology

    > How do labelled RF training, EVA generation, and Hill biophysics connect?

    <br>
    **Layout** (square):
    <br>
    <br>
    - Top = green EVA bar (horizontal: Model → Latent → Yield → **De Novo**),
    <br>
    - Bottom-left = grey Supervised RF,
    <br>
    - Bottom-right = in silico biophysics (directly beside RF).
    <br>
    <br>
    **Arrows**:
    <br>
    <br>
    - De Novo → **Thermal Sweep** (edge ends inside that box),
    <br>
    - Rfam Positive → Feature Extraction, plus branch → Thermal Sweep,
    <br>
    - Feature Extraction = **K-mers + Static Thermo Features**.
    """
    )

    workflow_png = fig_dir / "workflow_architecture.png"
    try:
        trf.export_workflow_architecture_png(workflow_png)
        _workflow_export_note = f"Saved → `{workflow_png}`"
    except Exception as exc:  # noqa: BLE001
        _workflow_export_note = (
            f"PNG export skipped ({type(exc).__name__}: {exc}). "
            f"Expected path → `{workflow_png}`."
        )

    mo.vstack([workflow_intro, workflow_diagram, mo.md(_workflow_export_note)])
    return


@app.cell
def _(fig_dir, mo, trf):
    """Thermoswitch mechanism schematic — basal lock vs heat-shock open."""
    fig_switch = trf.plot_thermoswitch_mechanism()
    switch_png = trf.save_fig(
        fig_switch, fig_dir / "thermoswitch_mechanism_rbs_lock.png"
    )

    _sec_switch = mo.md(
        rf"""
    ## RNA thermoswitch mechanism

    > How does a temperature rise turn translation from off to on?

    <br>
    **Actors**
    <br>
    <br>
    - RNA backbone (ribbon),
    <br>
    - RBS docking zone (orange),
    <br>
    - AUG start codon (purple; 3′ of RBS),
    <br>
    - Ribosome (two interlocking ovals).
    <br>
    <br>
    **37 °C — basal lock**
    <br>
    Stem–loop sequesters RBS + AUG → docking blocked → translation repressed.
    <br>
    <br>
    **42 °C — heat shock ($\Delta T$)**
    <br>
    Stem melts → RBS exposed (AUG downstream) → ribosome docks → translation activated.
    <br>
    <br>
    Saved → `notebooks/figures/thesis_figures/{switch_png.name}`.
    """
    )
    mo.vstack([_sec_switch, fig_switch])
    return


@app.cell
def _(fig_dir, mo, trf):
    """Figure 3 — RF StratifiedGroupKFold (5-fold) schematic."""
    rf_intro = mo.md(
        r"""
    ## Random Forest ensemble

    > How does the matched panel train trees under honest group holdout?

    <br>
    **What this shows**:
    <br>
    <br>
    - Matched dataset $N=2396$ split into **5 grouped folds**,
    <br>
    - partition by `rfam_acc` / RefSeq assembly — **no homology overlap** train↔val,
    <br>
    - each rotation: **4 folds train** ($T=200$ trees), **1 fold OOD validation**,
    <br>
    - repeat $k=1\ldots5$; aggregate out-of-fold $\hat{y}$ and ROC curves.
    """
    )

    rf_mermaid_src = trf.RF_ENSEMBLE_CV_MERMAID

    rf_ensemble_diagram = mo.mermaid(rf_mermaid_src)

    rf_png = fig_dir / "rf_ensemble_train_test.png"
    _rf_export_note = ""
    try:
        trf.export_rf_grouped_cv_png(rf_png)
        _rf_export_note = (
            f"Saved → `{rf_png}` "
            f"({rf_png.stat().st_size // 1024} KB, mermaid-cli scale 3)."
        )
    except Exception as exc:  # noqa: BLE001
        _rf_export_note = (
            f"PNG export skipped ({type(exc).__name__}: {exc}). "
            f"Mermaid still renders in-notebook. Expected path → `{rf_png}`."
        )

    mo.vstack([rf_intro, rf_ensemble_diagram, mo.md(_rf_export_note)])
    return


@app.cell
def _(fig_dir, mo, trf):
    """Figure 3b — RF ensemble schematic (version 2 potential)."""
    rf_v2_intro = mo.md(
        r"""
    ## Random Forest ensemble · version 2 (potential)

    > How does the non-circular $p=92$ feature vector feed a mean-probability ensemble under grouped CV?

    <br>
    **What this draft adds over v1**:
    <br>
    <br>
    - explicit **$p=92$** feature inventory (80 k-mers + static physics + composition + spacing),
    <br>
    - melting scalars ($T_m$, Hill, $Z$) marked **excluded from $X$**,
    <br>
    - mean soft-vote aggregation $\hat{y}(x)=\frac{1}{T}\sum h_t(x)$ instead of majority-vote cartoon,
    <br>
    - OOF headline metric: **StratifiedGroupKFold ROC-AUC $=0.277$**.
    """
    )

    rf_v2_diagram = mo.mermaid(trf.RF_ENSEMBLE_CV_MERMAID_V2)

    rf_v2_png = fig_dir / "rf_ensemble_train_test_v2_potential.png"
    _rf_v2_export_note = ""
    try:
        trf.export_rf_grouped_cv_v2_png(rf_v2_png)
        _rf_v2_export_note = (
            f"Saved → `{rf_v2_png}` "
            f"({rf_v2_png.stat().st_size // 1024} KB, mermaid-cli scale 3)."
        )
    except Exception as exc:  # noqa: BLE001
        _rf_v2_export_note = (
            f"PNG export skipped ({type(exc).__name__}: {exc}). "
            f"Mermaid still renders in-notebook. Expected path → `{rf_v2_png}`."
        )

    mo.vstack([rf_v2_intro, rf_v2_diagram, mo.md(_rf_v2_export_note)])
    return


@app.cell
def _(GT, loc, md, mo, pd, px, style):
    """Methods blanks filled: RF hyperparams, EVA triage, SciPy Hill fit."""
    methods_intro = mo.md(
        r"""
    ## Methods parameters (thesis 3.2.2 / 3.3.2 / 3.4.2)

    Exact numerical settings pulled from this repo — ready to paste into Methods.

    > What were the concrete RF, EVA triage, and SciPy Hill-fit parameters?
    """
    )

    # --- 3.2.2 Random Forest ---
    rf_blurb = mo.md(
        r"""
    ### 3.2.2 Random Forest parameters

    The final ensemble utilized $T = 200$ trees with unconstrained maximum depth
    (`max_depth=None`), `random_state=42`, and `n_jobs=-1`, evaluated via
    **5-fold stratified group cross-validation**
    (`StratifiedGroupKFold`, shuffle=True, grouped by `rfam_acc` /
    `REFSEQ:{{assembly}}`).

    <br>
    **Source**: `src/thermo_sim/rf_posthoc.py`, notebooks 06/07
    (`n_estimators=200`; honesty metric = GroupKFold AUC).
    <br>
    *(A separate shallow RF path in `thermo_classifier.py` uses
    $T=300$, `max_depth=4` — not the non-circular honesty ladder.)*
    """
    )

    # --- 3.3.2 EVA triage ---
    eva_blurb = mo.md(
        r"""
    ### 3.3.2 Generative triage thresholds

    EVA generated a **raw quota of $N=2000$** accepted sequences
    (pilot 512 + stream top-up; soft-drop kept the run alive).
    Mac yield triage retained **105 / 2000 = 5.25%**.

    <br>
    **Soft-drop (GPU orchestrator — “fatal structural anomalies”)** —
    drop a sequence if any of:
    <br>
    <br>
    - non-AUGC / empty / whitespace,
    <br>
    - length outside $[40, 600]$,
    <br>
    - mono-nt or dimer fraction $\ge 0.85$,
    <br>
    - unique 3-mer ratio $< 0.05$ (when $L \ge 80$),
    <br>
    - $\ge 3$ identical copies or near-duplicates at identity $\ge 0.98$.
    <br>
    <br>
    **Yield gate (Mac triage — switch-like + novel)**:
    <br>
    <br>
    $$
    Z \le -2
    \quad\wedge\quad
    \Delta P_{\mathrm{RBS}} > 0
    \quad\wedge\quad
    E_{\mathrm{Rfam}} > 10^{-3}
    $$
    <br>
    ($E_{\mathrm{Rfam}} = \min(\text{blastn}, \text{nhmmer})$; no-hit $\Rightarrow +\infty$, passes novelty.)
    <br>
    <br>
    **Source**: `eva_quality.py`, `eva_yield_ratio.py`, `cluster/macleod_log.md`.
    """
    )

    # --- 3.4.2 SciPy Hill fit ---
    hill_blurb = mo.md(
        r"""
    ### 3.4.2 SciPy Hill-fit parameters

    Melting curves were fit with `scipy.optimize.curve_fit` on the Hill sigmoid
    in `fit_hill_curve` (`thermo_common.py`).
    Because **box bounds are supplied**, SciPy selects **Trust Region Reflective
    (`method='trf'`)** — not Levenberg–Marquardt.

    <br>
    **Initial guess $p_0$**:
    <br>
    <br>
    $$
    p_0 = \big(y_{\min},\; y_{\max},\; T_{\mathrm{mid}},\; 2.0\big)
    $$
    <br>
    **Bounds** $(\mathrm{lower}, \mathrm{upper})$ for
    $(y_{\min},\, y_{\max},\, T_m,\, n_{\mathrm{H}})$:
    <br>
    <br>
    - $y_{\min}$: $[\min(y)-0.5,\; \max(y)+0.5]$
    <br>
    - $y_{\max}$: $[\min(y),\; \max(y)+0.5]$
    <br>
    - $T_m$: $[5,\; 95]^\circ\mathrm{C}$
    <br>
    - $n_{\mathrm{H}}$: $[0.1,\; 20]$
    <br>
    <br>
    `maxfev=10\,000`. Flat curves ($\Delta y < 10^{-6}$) skip optimization
    (`fit_status="flat"`).
    """
    )

    params_df = pd.DataFrame(
        [
            {
                "Section": "3.2.2 RF",
                "Parameter": "n_estimators (T)",
                "Value": "200",
            },
            {
                "Section": "3.2.2 RF",
                "Parameter": "max_depth",
                "Value": "None (unconstrained)",
            },
            {
                "Section": "3.2.2 RF",
                "Parameter": "CV",
                "Value": "StratifiedGroupKFold, 5 folds",
            },
            {
                "Section": "3.2.2 RF",
                "Parameter": "random_state / n_jobs",
                "Value": "42 / -1",
            },
            {
                "Section": "3.3.2 EVA",
                "Parameter": "Raw generation volume",
                "Value": "2000 accepted (before yield gate)",
            },
            {
                "Section": "3.3.2 EVA",
                "Parameter": "Soft-drop length",
                "Value": "[40, 600] nt",
            },
            {
                "Section": "3.3.2 EVA",
                "Parameter": "Soft-drop complexity",
                "Value": "mono/dimer ≥0.85; 3-mer ratio <0.05",
            },
            {
                "Section": "3.3.2 EVA",
                "Parameter": "Yield gate",
                "Value": "Z≤−2 ∧ ΔP_RBS>0 ∧ E_Rfam>1e−3",
            },
            {
                "Section": "3.3.2 EVA",
                "Parameter": "Passers",
                "Value": "105 / 2000 (5.25%)",
            },
            {
                "Section": "3.4.2 SciPy",
                "Parameter": "Optimizer",
                "Value": "curve_fit → trf (bounds set)",
            },
            {
                "Section": "3.4.2 SciPy",
                "Parameter": "p0",
                "Value": "(ymin, ymax, T_mid, 2.0)",
            },
            {
                "Section": "3.4.2 SciPy",
                "Parameter": "Tm / nH bounds",
                "Value": "Tm∈[5,95]; nH∈[0.1,20]",
            },
            {
                "Section": "3.4.2 SciPy",
                "Parameter": "maxfev",
                "Value": "10000",
            },
        ]
    )

    params_table = (
        GT(params_df)
        .tab_header(
            title=md("**Methods parameter card**"),
            subtitle=md("RF · EVA triage · SciPy Hill fit — from this repository"),
        )
        .cols_align(align="left")
        .tab_options(
            table_width="100%",
            table_font_size=px(15),
            heading_title_font_size=px(22),
            heading_title_font_weight="bold",
            column_labels_font_weight="bold",
            column_labels_background_color="#E8EEF2",
            data_row_padding=px(8),
            table_border_top_width=px(2),
            table_border_bottom_width=px(2),
            row_striping_background_color="#F7F9FB",
        )
        .tab_style(
            style=style.text(weight="bold"),
            locations=loc.body(columns="Section"),
        )
        .opt_row_striping()
        .opt_stylize(style=1, color="gray")
    )

    paste_ready = mo.md(
        r"""
    ### Paste-ready Methods sentences

    **3.2.2:** The final ensemble utilized $T = 200$ trees with unconstrained
    maximum depth, evaluated via 5-fold stratified group cross-validation.
    <br>
    <br>
    **3.3.2:** From 2000 soft-drop-accepted EVA sequences, yield triage
    ($Z \le -2$, $\Delta P_{\mathrm{RBS}} > 0$, $E_{\mathrm{Rfam}} > 10^{-3}$)
    retained $N = 105$; soft-drop removed non-AUGC, length-outliers outside
    $[40,600]$, and low-complexity collapses (mono/dimer $\ge 0.85$;
    unique 3-mer ratio $< 0.05$).
    <br>
    <br>
    **3.4.2:** Hill parameters were obtained with `scipy.optimize.curve_fit`
    under Trust Region Reflective (`trf`) using data-driven $p_0$ and box
    bounds on $(y_{\min}, y_{\max}, T_m \in [5,95], n_{\mathrm{H}} \in [0.1,20])$,
    `maxfev=10000`.
    """
    )

    mo.vstack(
        [
            methods_intro,
            rf_blurb,
            eva_blurb,
            hill_blurb,
            params_table,
            paste_ready,
        ]
    )
    return


@app.cell
def _(data_dir, mo, trf):
    """Load Results sidecars (diagnostics, post-hoc, OOF curves)."""
    diag = trf.load_json(data_dir / "rf_noncircular_diagnostics.json")
    post = trf.load_json(data_dir / "rf_posthoc_report.json")
    curves = trf.load_json(data_dir / "rf_noncircular_oof_curves.json")

    results_intro = mo.md(
        r"""
    # Results figures
    > Does the supervised RF generalise, or only memorise family k-mers —

    <br>
    and do EVA passers melt like thermoswitches?

    <br>
    **Sidecars**: `rf_noncircular_diagnostics.json`,
    <br>
    `rf_posthoc_report.json`,
    <br>
    `rf_noncircular_oof_curves.json`.
    """
    )
    mo.vstack([results_intro])
    return curves, diag, post


@app.cell
def _(curves, fig_dir, mo, trf):
    """4.1.2 — leaky vs honest ROC overlay."""
    fig_roc = trf.plot_cv_collapse_roc(curves)
    roc_png = trf.save_fig(fig_roc, fig_dir / "04_1_2_cv_group_holdout_roc.png")

    _sec_roc = mo.md(
        rf"""
    ## Supervised baseline — CV vs group holdout

    > Does random-split AUC prove a thermoswitch detector?

    <br>
    **Leaky** StratifiedKFold AUC $= {curves["auc_skf"]:.3f}$
    <br>
    hugs the top-left corner.
    <br>
    <br>
    **Honest** StratifiedGroupKFold AUC $= {curves["auc_sgkf"]:.3f}$
    <br>
    collapses onto the chance diagonal.
    <br>
    <br>
    Saved → `notebooks/figures/thesis_figures/{roc_png.name}`.
    """
    )
    mo.vstack([_sec_roc, fig_roc])
    return


@app.cell
def _(PROJECT_ROOT, diag, fig_dir, mo, trf):
    """Feature attribution — naive raw MFE vs strict MFE/nt (matched blocks)."""
    naive_path = (
        PROJECT_ROOT / "data" / "processed" / "rf_naive_raw_mfe_permutation.json"
    )
    naive_diag = trf.load_json(naive_path)
    fig_imp = trf.plot_naive_vs_strict_attribution(naive_diag, diag)
    imp_png = trf.save_fig(fig_imp, fig_dir / "04_1_3_grouped_permutation.png")

    _tri_n = naive_diag["grouped_permutation_importance"]["groups"]["trinucleotides"]
    _tri = diag["grouped_permutation_importance"]["groups"]["trinucleotides"]
    _bio_n = naive_diag["grouped_permutation_importance"]["groups"]["static_biophysics"]
    _bio = diag["grouped_permutation_importance"]["groups"]["static_biophysics"]

    _sec_imp = mo.md(
        rf"""
    ## Feature attribution — naive vs strict

    > Same feature blocks — does raw MFE vs MFE/nt change what the forest uses?

    <br>
    **Matched design**: identical $k$-mers, composition, SD–AUG, static physics.
    <br>
    **Only difference**: naive uses raw MFE; strict uses intensive MFE/nt.
    <br>
    <br>
    **Panel A — Naive** (raw MFE):
    <br>
    Trinucleotides mean AUC drop $= {float(_tri_n["mean_auc_drop"]):.3f}$;
    <br>
    static biophysics $\approx {float(_bio_n["mean_auc_drop"]):.4f}$.
    <br>
    <br>
    **Panel B — Strict** (MFE/nt):
    <br>
    Trinucleotides mean AUC drop $= {float(_tri["mean_auc_drop"]):.3f}$;
    <br>
    static biophysics $\approx {float(_bio["mean_auc_drop"]):.4f}$.
    <br>
    <br>
    Motif dominance is stable to the MFE vs MFE/nt choice.
    <br>
    Saved → `notebooks/figures/thesis_figures/{imp_png.name}`.
    """
    )
    mo.vstack([_sec_imp, fig_imp])
    return


@app.cell
def _(curves, fig_dir, mo, post, trf):
    """4.1.4 — OOF confidence histogram."""
    fig_conf = trf.plot_oof_confidence_hist(curves)
    conf_png = trf.save_fig(fig_conf, fig_dir / "04_1_4_oof_confidence_hist.png")
    _bins = post["bins"]

    _sec_conf = mo.md(
        rf"""
    ## Confidence calibration — OOD transfer

    > Can $\hat{{y}}$ gate wet-lab discovery under family holdout?

    <br>
    **Low** ($\hat{{y}} \le 0.20$): $n = {_bins["low_le_0.20"]}$.
    <br>
    **High** ($\hat{{y}} \ge 0.80$): $n = {_bins["high_ge_0.80"]}$.
    <br>
    <br>
    Near-total uncertainty out-of-family — not a hard discovery gate.
    <br>
    Saved → `notebooks/figures/thesis_figures/{conf_png.name}`.
    """
    )
    mo.vstack([_sec_conf, fig_conf])
    return


@app.cell
def _(PROJECT_ROOT, fig_dir, mo, pd, post, trf):
    """4.2.1 — checklist UpSet + Vienna–NUPACK MFE vs Tm concordance."""
    from scipy.stats import spearmanr

    fused_path = (
        PROJECT_ROOT / "data" / "processed" / "fused_features_refseq_dynamic.csv"
    )
    cols = [
        "label",
        "viennarna_MFE",
        "nupack_MFE",
        "viennarna_Tm",
        "nupack_Tm",
        "viennarna_hill_coeff",
        "viennarna_amplitude",
        "viennarna_P_open_RBS_37",
    ]
    panel = pd.read_csv(fused_path, usecols=cols)

    fig_upset = trf.plot_checklist_upset(panel)
    upset_png = trf.save_fig(fig_upset, fig_dir / "04_2_1_checklist_upset.png")

    _mfe = panel[["viennarna_MFE", "nupack_MFE"]].dropna()
    rs_mfe = float(spearmanr(_mfe["viennarna_MFE"], _mfe["nupack_MFE"]).statistic)
    rs_tm = float(
        post["spearman_panel_wide_primary"]["pairs"]["viennarna_Tm_vs_nupack_Tm"]["r_s"]
    )
    fig_tm = trf.plot_tm_concordance(panel, r_s_mfe=rs_mfe, r_s_tm=rs_tm)
    tm_png = trf.save_fig(fig_tm, fig_dir / "04_2_1_vienna_nupack_tm.png")
    _check = post["visual_checklist"]

    _sec_posthoc = mo.md(
        rf"""
    ## Post-hoc phenotype and engine concordance

    > Static MFE agrees across engines —

    <br>
    does melting $T_m$ also agree?

    <br>
    **All-four checklist intersection**: $n = {_check["all_four_pass"]}$.
    <br>
    <br>
    **Panel A — Static MFE**:
    - $r_s \approx {rs_mfe:.3f}$
    - tight diagonal on $y=x$
    <br>
    **Panel B — Dynamic $T_m$**:
    - $r_s \approx {rs_tm:.3f}$
    - near-zero rank concordance (cloud)
    <br>
    <br>
    Saved → `{upset_png.name}`, `{tm_png.name}`.
    """
    )
    posthoc_view = mo.vstack([_sec_posthoc, fig_upset, fig_tm])
    return panel, posthoc_view


@app.cell
def _(PROJECT_ROOT, fig_dir, mo, np, pd, trf):
    """4.2.2 — EVA attrition funnel (2000 → 105)."""
    pilot = pd.read_csv(
        PROJECT_ROOT / "data" / "processed" / "eva_pilot" / "yield_ratio_sequences.csv"
    )
    stream = pd.read_csv(
        PROJECT_ROOT / "data" / "processed" / "eva_stream" / "yield_ratio_sequences.csv"
    )
    eva = pd.concat([pilot, stream], ignore_index=True)
    z = pd.to_numeric(eva["viennarna_mfe_zscore"], errors="coerce")
    dp = pd.to_numeric(eva["viennarna_delta_P_RBS"], errors="coerce")
    e = pd.to_numeric(eva["E_Rfam"], errors="coerce")
    novel = np.isinf(e.to_numpy()) | (e.to_numpy() > 1e-3)
    g_z = z <= -2
    g_dp = dp > 0
    stages = [
        ("Soft-drop accepted", int(len(eva))),
        ("MFE Z-score ≤ −2", int(g_z.sum())),
        ("+ ΔP_RBS > 0", int((g_z & g_dp).sum())),
        ("+ E_Rfam > 10⁻³  (yield gate)", int((g_z & g_dp & novel).sum())),
    ]
    fig_funnel = trf.plot_eva_attrition_funnel(stages)
    funnel_png = trf.save_fig(fig_funnel, fig_dir / "04_2_2_eva_attrition_funnel.png")

    _n0, _n1, _n2, _n3 = (s[1] for s in stages)
    _sec_funnel = mo.md(
        rf"""
    ## Generative viability — EVA triage yield

    > Of 2,000 soft-drop-accepted EVA sequences, how many clear the biology gates?

    <br>
    **Attrition**
    <br>
    <br>
    Soft-drop cohort → {_n0:,}
    <br>
    $Z \le -2$ → {_n1:,}
    <br>
    $+\;\Delta P_{{\mathrm{{RBS}}}} > 0$ → {_n2:,}
    <br>
    $+\;E_{{\mathrm{{Rfam}}}} > 10^{{-3}}$ → **{_n3:,}**
    <br>
    <br>
    **Overall yield**: ${_n3}/{_n0} = {_n3 / _n0:.2%}$.
    <br>
    Saved → `notebooks/figures/thesis_figures/{funnel_png.name}`.
    """
    )
    mo.vstack([_sec_funnel, fig_funnel])
    return


@app.cell
def _(PROJECT_ROOT, fig_dir, mo, panel, pd, trf):
    """4.3 — Hill melting overlay (Rfam selected vs EVA sweeps)."""
    sweeps_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "eva_characterization"
        / "eva_temp_sweeps.csv"
    )
    eva_curves = pd.read_csv(sweeps_path) if sweeps_path.exists() else pd.DataFrame()
    rfam_params = trf.rfam_hill_param_table(panel)
    fig_hill = trf.plot_hill_melting_overlay(rfam_params, eva_curves)
    hill_png = trf.save_fig(fig_hill, fig_dir / "04_3_hill_melting_overlay.png")

    _sec_hill = mo.md(
        rf"""
    ## Dynamic validation — in silico thermal sweeps

    > Do EVA passers show sharp RBS opening across 37–42 °C like selected Rfam?

    <br>
    **Rfam panel**: high-$n_H$ / high-$\Delta\theta$ reconstructions ($n = {len(rfam_params)}$).
    <br>
    **EVA**: measured 1 °C sweeps on yield-gated passers
    ($n_{{\mathrm{{seq}}}} = {0 if eva_curves.empty else eva_curves["record_id"].nunique()}$).
    <br>
    <br>
    Flat EVA trajectories vs steeper selected Rfam mean = design intermediates.
    <br>
    Saved → `notebooks/figures/thesis_figures/{hill_png.name}`.
    """
    )
    mo.vstack([_sec_hill, fig_hill])
    return


@app.cell
def _(GT, PROJECT_ROOT, fig_dir, loc, md, mo, px, style, trf):
    """4.3 — Table 3 thermo/Hill benchmark (Rfam, EVA all, Tier-1 A/B)."""
    thermo_summary_df = trf.build_thermo_benchmark_tables(PROJECT_ROOT)

    thermo_table = (
        GT(thermo_summary_df)
        .tab_header(
            title=md(
                "**Table 3: Thermodynamic and Hill Parameter Summary of "
                "Natural Controls vs. De Novo Candidates**"
            ),
            subtitle=md(
                "Rfam positives · EVA yield-gated (all) · Tier-1 Candidates A/B"
            ),
        )
        .cols_align(align="left", columns=["Cohort / Candidate"])
        .cols_align(
            align="center",
            columns=[c for c in thermo_summary_df.columns if c != "Cohort / Candidate"],
        )
        .tab_options(**trf.docs_table_options(px))
        .tab_style(
            style=style.text(weight="bold", size=px(28)),
            locations=loc.body(columns="Cohort / Candidate"),
        )
        .tab_source_note(
            source_note=md(
                "$\\Delta G_{\\mathrm{density}}$ = Vienna MFE / nt (kcal/mol/nt) · "
                "$P_{\\mathrm{open}}(37^\\circ\\mathrm{C})$ = $y_{\\min}$ · "
                "$\\Delta P_{\\mathrm{RBS}} = P_{55}-P_{37}$ · "
                "$T_m$, $h$ from Vienna Hill fits · "
                "PNG: 4× zoom + 96 DPI for Docs paste."
            )
        )
        .opt_row_striping()
        .opt_table_outline(style="solid", width=px(2), color="#222222")
        .opt_stylize(style=1, color="gray")
    )

    thermo_png = fig_dir / "thermo_hill_benchmark_google_docs.png"
    _thermo_export_note = ""
    try:
        # Same Docs recipe as cohort_map / compositional ablation.
        thermo_table.gtsave(
            str(thermo_png),
            zoom=4.0,
            vwidth=1200,
            expand=24,
        )
        trf.fit_docs_png_width(thermo_png, target_width=4592, dpi=96)
        _thermo_export_note = (
            f"Saved Docs-ready PNG → `{thermo_png}` "
            f"({thermo_png.stat().st_size // 1024} KB, 4× zoom, 96 DPI, width=4592)."
        )
    except Exception as exc:  # noqa: BLE001 — Chrome may be missing in some envs
        _thermo_export_note = (
            f"PNG export skipped ({type(exc).__name__}: {exc}). "
            "Table still renders in-notebook."
        )

    thermo_html = mo.Html(
        f"""
        <div style="zoom:1.25; transform-origin: top left; max-width: 100%;">
          {thermo_table.as_raw_html()}
        </div>
        """
    )

    _sec_thermo = mo.md(
        r"""
    ## Biophysical & thermodynamic benchmark

    > How do natural Rfam positives and the full EVA yield-gated cohort compare
    to the two Tier-1 candidates on thermodynamic and Hill parameters?

    <br>
    **What Table 3 shows**:
    <br>
    <br>
    - Rfam Positive Controls ($N=1{,}198$),
    <br>
    - EVA Yield-Gated (All) ($N=105$),
    <br>
    - Tier-1 A (`eva_sample_315`),
    <br>
    - Tier-1 B (`eva_sample_1858`).
    <br>
    <br>
    Columns: $\Delta G_{\mathrm{density}}$, $P_{\mathrm{open}}(37^\circ\mathrm{C})$ ($y_{\min}$),
    <br>
    $P_{\mathrm{open}}(42^\circ\mathrm{C})$, $\Delta P_{\mathrm{RBS}}$, activation $T_m$, Hill $h$.
    """
    )

    mo.vstack(
        [
            _sec_thermo,
            thermo_html,
            mo.md(_thermo_export_note),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
