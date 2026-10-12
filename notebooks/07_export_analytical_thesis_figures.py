"""07 · Export analytical thesis figures.

Publication panels for the Results narrative only (ROC, permutation, funnel,
Hill overlay) — no flowchart schematics.

Run:
    uv run marimo edit notebooks/07_export_analytical_thesis_figures.py
"""

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full", app_title="07 · Analytical thesis figures")


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
    # 07 · Analytical thesis figures

    **What we export:** Results panels only (ROC, permutation, funnel, Hill) —
    no flowchart schematics.

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
                "Length/%GC-only AUCs $< 0.50$ on the matched panel. "
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

    > Did length/%GC-only forests still separate classes after Hungarian matching?

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
    Near-total uncertainty out-of-family under StratifiedGroupKFold.
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
