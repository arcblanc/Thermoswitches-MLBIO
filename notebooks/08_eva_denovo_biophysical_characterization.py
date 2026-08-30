"""Marimo App 8: EVA de novo biophysical characterization.

Compares 105 yield-gated EVA thermoswitch candidates to Rfam / RefSeq controls.

Run:
    uv run marimo edit notebooks/08_eva_denovo_biophysical_characterization.py
"""

import marimo

__generated_with = "0.24.0"
app = marimo.App(
    width="full",
    app_title="EVA de novo biophysical characterization",
)


@app.cell
def _():
    import sys
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    PROJECT_ROOT = Path.cwd().resolve()
    if not (PROJECT_ROOT / "src").exists():
        PROJECT_ROOT = PROJECT_ROOT.parent
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

    from thermo_sim.eva_denovo_characterization import (
        SCORE_WEIGHTS,
        add_induction_ratio,
        backfill_eva_p_open,
        characterization_paths,
        characterize_mutation_trio,
        checklist_by_cohort,
        cohort_stat_tests,
        derivative_profiles,
        embed_structure_manifold,
        engine_concordance,
        export_leads,
        flag_software_fragile,
        gate_survival_funnel,
        hill_fits_from_sweeps,
        load_control_panel,
        load_eva_passers,
        load_prototype_overlay,
        merge_eva_fits,
        plot_before_after_positional_heatmaps,
        plot_derivative_curves,
        plot_engine_parity,
        plot_gate_survival_funnel,
        plot_induction_ratio_violins,
        plot_leakiness,
        plot_mutation_rescue_curves,
        plot_nh_vs_tm,
        plot_novelty_vs_nh,
        plot_positional_unpairing_heatmap,
        plot_ribbon_curves,
        plot_structure_manifold,
        plot_violins,
        positional_unpairing_matrix,
        rank_leads,
        ribbon_from_hill_params,
        ribbon_from_sweep_table,
        run_or_load_eva_sweeps,
        run_scaffold_gc_clamp_poc,
        save_figure,
        sd_span_for_sequence,
        structure_distance_matrix,
        sweep_temps,
        write_checklist_summary,
    )

    plt.rcParams.update({"figure.dpi": 120, "savefig.bbox": "tight"})
    return (
        PROJECT_ROOT,
        Path,
        SCORE_WEIGHTS,
        add_induction_ratio,
        backfill_eva_p_open,
        characterization_paths,
        characterize_mutation_trio,
        checklist_by_cohort,
        cohort_stat_tests,
        derivative_profiles,
        embed_structure_manifold,
        engine_concordance,
        export_leads,
        flag_software_fragile,
        gate_survival_funnel,
        hill_fits_from_sweeps,
        load_control_panel,
        load_eva_passers,
        load_prototype_overlay,
        merge_eva_fits,
        mo,
        np,
        pd,
        plot_before_after_positional_heatmaps,
        plot_derivative_curves,
        plot_engine_parity,
        plot_gate_survival_funnel,
        plot_induction_ratio_violins,
        plot_leakiness,
        plot_mutation_rescue_curves,
        plot_nh_vs_tm,
        plot_novelty_vs_nh,
        plot_positional_unpairing_heatmap,
        plot_ribbon_curves,
        plot_structure_manifold,
        plot_violins,
        positional_unpairing_matrix,
        rank_leads,
        ribbon_from_hill_params,
        ribbon_from_sweep_table,
        run_or_load_eva_sweeps,
        run_scaffold_gc_clamp_poc,
        save_figure,
        sd_span_for_sequence,
        structure_distance_matrix,
        sweep_temps,
        write_checklist_summary,
    )


@app.cell
def _(mo):
    def cell_status(label: str, ok: bool, detail: str) -> mo.Html:
        """Render a one-line pass/fail badge for a pipeline checkpoint."""
        icon = "✅" if ok else "❌"
        return mo.md(f"{icon} **{label}** — {detail}")

    return (cell_status,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # EVA de novo biophysical characterization

    We take **105** EVA candidates that already passed yield triage
    (**31 pilot** + **74 stream**) and ask a harder question:

    > Do they melt like real thermoswitches, or only look good under one software score?

    ---

    ### 1. Yield Triage Baseline (Admissions Gate)

    **Definition:**
    <br>
    Filter requiring:
    <br>
    - thermodynamic over-stabilization ($Z \le -2$),
    <br>
    - net unpairing on heating ($\Delta P_{\mathrm{RBS}} > 0$),
    <br>
    - covariance alignment ($E_{\mathrm{Rfam}} > 10^{-3}$).

    <br>
    **What a positive shows:**
    <br>
    <br>
    - evolved-like structural core,
    <br>
    - sequestered RBS at $37^\circ\mathrm{C}$,
    <br>
    - baseline stability beyond random sequence noise.

    <br>
    **Significance:**
    <br>
    Coarse sieve — discards ~$99\%$ of non-functional generative hallucinations
    before heavy biophysical sweeps.

    ---

    ### 2. Core Cooperative Melting Checks

    > Are the survivors sharp, heat-shock-timed, and locked at rest?

    <br>
    **Hill snap ($n_{\mathrm{H}} > 1.5$)**
    <br>
    <br>
    **Definition:** fit unzipping to a Hill sigmoid:
    <br>

    $$\theta(T) = \theta_{\mathrm{min}} + (\theta_{\mathrm{max}} - \theta_{\mathrm{min}})
    \frac{T^{n_{\mathrm{H}}}}{T_{\mathrm{m}}^{n_{\mathrm{H}}} + T^{n_{\mathrm{H}}}}$$

    <br>
    **What a positive shows:**
    <br>
    - sharp, cooperative, switch-like unfold ($n_{\mathrm{H}} > 1.5$),
    <br>
    - not a non-specific linear ramp ($n_{\mathrm{H}} \le 1.0$).

    <br>
    **Significance:**
    <br>
    Distinguishes allosteric molecular switches from generic thermal denaturing.

    <br><br>
    **Heat-shock midpoint ($T_{\mathrm{m}} \in [42, 45]^\circ\mathrm{C}$)**
    <br>
    <br>
    **Definition:**
    <br>
    Temperature where half the RBS is exposed.

    <br>
    **What a positive shows:**
    <br>
    Unzipping aligns with physiological bacterial heat-shock ($42$–$45^\circ\mathrm{C}$).

    <br>
    **Significance:**
    <br>
    - activates under stress,
    <br>
    - not at standard growth,
    <br>
    - not frozen until lethal heat ($>55^\circ\mathrm{C}$).

    <br><br>
    **Locked $37^\circ\mathrm{C}$ baseline ($P_{\mathrm{open}}^{37^\circ\mathrm{C}} \le 0.20$)**
    <br>
    <br>
    **Definition:**
    <br>
    Fraction of RBS unpairing at resting temperature ($37^\circ\mathrm{C}$).

    <br>
    **What a positive shows:**
    <br>
    Shine–Dalgarno inaccessible to the 30S subunit at basal body temperature.

    <br>
    **Significance:**
    <br>
    Prevents translational leakiness during normal growth (cytotoxic burden).

    ---

    ### 3. Six ML & SynBio Diagnostic Views

    > After admissions + melting gates — what else must be true?

    <br>
    **1. Positional melting heatmaps**
    <br>
    <br>
    **Definition:**
    <br>
    Base-by-base unpairing vs temperature ($30^\circ\mathrm{C} \to 60^\circ\mathrm{C}$ by position $1 \to L$).

    <br>
    **What a positive shows:**
    <br>
    - melt localized to the RBS hairpin,
    <br>
    - scaffold / anchor stems stay paired.

    <br>
    **Significance:**
    <br>
    Targeted modular switch — not global secondary-structure collapse.

    <br><br>
    **2. Translational induction ratio ($\mathrm{Fold\text{-}Change}_{\mathrm{ON}/\mathrm{OFF}}$)**
    <br>
    <br>
    **Definition:**
    <br>

    $$\mathrm{Fold\text{-}Change}
    = \frac{P_{\mathrm{open}}(42^\circ\mathrm{C})}{P_{\mathrm{open}}(37^\circ\mathrm{C})}$$

    <br>
    **What a positive shows:**
    <br>
    $\ge 5\times$–$10\times$ rise in translation-initiation potential.

    <br>
    **Significance:**
    <br>
    Maps biophysics → wet-lab signal-to-noise (GFP / luciferase reporters).

    <br><br>
    **3. Derivative transition sharpness ($\mathrm{d}\theta/\mathrm{d}T$)**
    <br>
    <br>
    **Definition:**
    <br>
    First derivative of opening fraction vs temperature.

    <br>
    **What a positive shows:**
    <br>
    Narrow, high-amplitude peak centered in $42$–$45^\circ\mathrm{C}$.

    <br>
    **Significance:**
    <br>
    Exact thermal window from repressed → activated.

    <br><br>
    **4. Structural fold diversity (Tree-Edit Distance MDS)**
    <br>
    <br>
    **Definition:**
    <br>
    2D map of pairwise MFE Tree Edit Distances (`RNAdistance`) across leads.

    <br>
    **What a positive shows:**
    <br>
    - multiple structural clusters,
    <br>
    - multi-stem / hairpin / bubble diversity,
    <br>
    - not one collapsed point.

    <br>
    **Significance:**
    <br>
    EVA samples diverse mechanisms — no structural mode collapse.

    <br><br>
    **5. Gate survival funnel**
    <br>
    <br>
    **Definition:**
    <br>
    Stepwise attrition through Gates 1→4.

    <br>
    **What a positive shows:**
    <br>
    Non-zero *de novo* yield clearing all four gates at once.

    <br>
    **Significance:**
    <br>
    Beats natural panels (≈$0\%$ simultaneous all-four survival on these strict filters).

    <br><br>
    **6. In silico mutational disruption & rescue**
    <br>
    <br>
    **Definition:**
    <br>
    Sweep trio:
    <br>
    - wild-type lead,
    <br>
    - RBS-stem mismatch (disrupt),
    <br>
    - compensatory base-pair (rescue).

    <br>
    **What a positive shows:**
    <br>
    - disrupt: $T_{\mathrm{m}} < 37^\circ\mathrm{C}$, higher leak,
    <br>
    - rescue: restores $42$–$45^\circ\mathrm{C}$ sigmoidal snap.

    <br>
    **Significance:**
    <br>
    Switching from engineered base-pairing — not composition artifacts.

    ---

    ### 4. Benchmark Overlay: The FourU Gold Standard

    **Role:**
    <br>
    **FourU** (*Salmonella agsA*) is the biological control overlay.
    <br>
    PrfA is **not** in this 4-sequence prototype panel.

    <br>
    **Significance:**
    <br>
    - well-characterized single-hairpin bacterial thermoswitch,
    <br>
    - SD paired against four consecutive uridines,
    <br>
    - benchmark for lead stability, steepness, and dynamic range vs a verified natural design.
    """)
    return


@app.cell
def _(mo):
    run_sweeps = mo.ui.switch(
        value=False,
        label="Run Vienna 1 °C sweeps on the 105 EVA passers (slow; uses cache when off)",
    )
    backfill_popen = mo.ui.switch(
        value=False,
        label="Backfill missing P_open(37/55 °C) with Vienna (no 100-shuffle Z re-run)",
    )
    run_heatmaps = mo.ui.switch(
        value=False,
        label="Compute positional unpairing heatmaps for top-3 leads (Vienna; slow)",
    )
    run_structure_embed = mo.ui.switch(
        value=False,
        label="Embed EVA MFE structures (MDS tree-edit; moderate)",
    )
    run_mutations = mo.ui.switch(
        value=False,
        label="Run in-silico disrupt/rescue sweeps on top-3 leads (Vienna; slow)",
    )
    mo.vstack(
        [run_sweeps, backfill_popen, run_heatmaps, run_structure_embed, run_mutations]
    )
    return (
        backfill_popen,
        run_heatmaps,
        run_mutations,
        run_structure_embed,
        run_sweeps,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## §1 Cohort ingestion

    Load `eva_pilot/top_candidates.fasta` ($N=31$),
    <br>
    `eva_stream/top_candidates.fasta` ($N=74$),
    <br>
    and the labelled panel from `fused_features_refseq_dynamic.csv` ($N=1{,}198$ Rfam /
    $N=1{,}198$ RefSeq).
    <br>
    Temperature sweeps default to **cache-first**:
    <br>
    reconstructed Hill ribbons for the 2,396-row panel; EVA Hill / $T_m$ appear after a sweep run or when `data/processed/eva_characterization/eva_temp_sweeps.csv` exists.
    """)
    return


@app.cell
def _(
    PROJECT_ROOT,
    Path,
    backfill_eva_p_open,
    backfill_popen,
    cell_status,
    characterization_paths,
    load_control_panel,
    load_eva_passers,
    load_prototype_overlay,
    mo,
):
    def fasta_n(path: Path) -> str:
        """Return a FASTA record count, or a missing-file note."""
        if not path.exists():
            return f"missing ({path})"
        n = sum(1 for line in path.read_text().splitlines() if line.startswith(">"))
        return f"{n} records @ {path.name}"

    paths = characterization_paths(PROJECT_ROOT)
    eva = load_eva_passers(paths)
    if backfill_popen.value:
        eva = backfill_eva_p_open(eva, run=True)
    controls = load_control_panel(paths["fused"])
    prototypes = load_prototype_overlay(paths)
    n_pilot = (
        int((eva["cohort"] == "EVA Pilot Top Passers").sum()) if not eva.empty else 0
    )
    n_stream = (
        int((eva["cohort"] == "EVA Stream Top Passers").sum()) if not eva.empty else 0
    )

    mo.vstack(
        [
            mo.md("### Pipeline checkpoints"),
            cell_status(
                "project root", (PROJECT_ROOT / "src").exists(), str(PROJECT_ROOT)
            ),
            cell_status(
                "pilot FASTA",
                paths["pilot_fasta"].exists(),
                fasta_n(paths["pilot_fasta"]),
            ),
            cell_status(
                "stream FASTA",
                paths["stream_fasta"].exists(),
                fasta_n(paths["stream_fasta"]),
            ),
            cell_status(
                "fused panel",
                paths["fused"].exists(),
                f"n={len(controls)}"
                if not controls.empty
                else f"missing ({paths['fused']})",
            ),
            cell_status(
                "EVA passers loaded",
                not eva.empty,
                f"n={len(eva)} (pilot={n_pilot}, stream={n_stream})",
            ),
            cell_status(
                "FourU prototype curves",
                not prototypes.empty,
                f"n={len(prototypes)} overlay rows",
            ),
        ]
    )
    return controls, eva, paths, prototypes


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## §2 Temperature sweeps and Hill sigmoid fits

    $$\theta(T) = \theta_{\mathrm{min}} + (\theta_{\mathrm{max}} - \theta_{\mathrm{min}})
    \frac{T^{n_{\mathrm{H}}}}{T_{\mathrm{m}}^{n_{\mathrm{H}}} + T^{n_{\mathrm{H}}}}$$

    EVA curves use the same SD-window unpaired series as `extract_vienna_features`
    (the 2,396-row panel). Fit failures with $R^2 < 0.95$ are flagged.
    Enable the sweep switch above to compute 30–60 °C at 1 °C for all 105 passers
    (cached under `data/processed/eva_characterization/`).
    """)
    return


@app.cell
def _(
    eva,
    hill_fits_from_sweeps,
    merge_eva_fits,
    mo,
    paths,
    run_or_load_eva_sweeps,
    run_sweeps,
):
    sweeps = run_or_load_eva_sweeps(
        eva, cache_path=paths["sweep_cache"], run=run_sweeps.value
    )
    fits = hill_fits_from_sweeps(sweeps)
    eva_fitted = merge_eva_fits(eva, fits)
    if not fits.empty:
        paths["hill_cache"].parent.mkdir(parents=True, exist_ok=True)
        fits.to_csv(paths["hill_cache"], index=False)
    n_converged = (
        int(fits["fit_converged"].sum())
        if not fits.empty and "fit_converged" in fits.columns
        else 0
    )
    n_fit = int(fits["record_id"].nunique()) if not fits.empty else 0
    sweep_note = (
        f"Loaded {len(sweeps)} sweep rows; Hill fits for {n_fit} sequences "
        f"({n_converged} with $R^2 \\ge 0.95$)."
        if not sweeps.empty
        else "No EVA temperature cache. Toggle **Run Vienna 1 °C sweeps** to fold the 105 passers, "
        "or keep cache-off to score leakiness / $Z$ / $\\Delta P_{\\mathrm{RBS}}$ only."
    )
    mo.md(sweep_note)
    return eva_fitted, sweeps


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## §3 Population diagnostics (EVA vs controls)

    Mann–Whitney $U$ and Kolmogorov–Smirnov tests: stream vs Rfam, stream vs RefSeq,
    stream vs pilot. Ribbons for the labelled panel are **reconstructed** from fitted
    Hill parameters; EVA ribbons use measured sweep points when the cache exists.
    """)
    return


@app.cell
def _(
    cohort_stat_tests,
    controls,
    eva_fitted,
    mo,
    np,
    paths,
    pd,
    plot_ribbon_curves,
    plot_violins,
    prototypes,
    ribbon_from_hill_params,
    ribbon_from_sweep_table,
    save_figure,
    sweep_temps,
    sweeps,
):
    panel = pd.concat(
        [df for df in (controls, eva_fitted) if df is not None and not df.empty],
        ignore_index=True,
        sort=False,
    )
    temps = np.asarray(sweep_temps(), dtype=float)
    control_ribbon = (
        ribbon_from_hill_params(controls, temps)
        if not controls.empty
        else pd.DataFrame()
    )
    eva_ribbon = ribbon_from_sweep_table(sweeps) if not sweeps.empty else pd.DataFrame()
    ribbon = pd.concat(
        [df for df in (control_ribbon, eva_ribbon) if not df.empty],
        ignore_index=True,
    )
    fig_ribbon = plot_ribbon_curves(ribbon, prototypes)
    fig_violin = plot_violins(panel)
    save_figure(fig_ribbon, paths["fig_dir"] / "ribbon_curves.png")
    save_figure(fig_violin, paths["fig_dir"] / "cohort_violins.png")
    tests = cohort_stat_tests(panel)
    mo.vstack(
        [
            fig_ribbon,
            fig_violin,
            mo.md("### Stream vs Rfam / RefSeq / pilot"),
            mo.ui.table(tests.round(4) if not tests.empty else pd.DataFrame()),
        ]
    )
    return (panel,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## §4 Multi-engine concordance and 5-plot audit

    Natural panel Spearman $r_s \approx 0.035$ (Vienna $T_m$ vs NUPACK $T_m$).
    Top-tier synthetic leads should keep $|\Delta T_m| \le 3^\circ\mathrm{C}$;
    sequences with $|\Delta T_m| > 5^\circ\mathrm{C}$ or switch-vs-coil disagreement
    are flagged **software-fragile**.
    """)
    return


@app.cell
def _(
    engine_concordance,
    flag_software_fragile,
    mo,
    panel,
    paths,
    plot_engine_parity,
    plot_leakiness,
    plot_nh_vs_tm,
    plot_novelty_vs_nh,
    save_figure,
):
    fig_box = plot_nh_vs_tm(panel)
    fig_leak = plot_leakiness(panel)
    fig_parity = plot_engine_parity(panel)
    fig_nov = plot_novelty_vs_nh(panel)
    save_figure(fig_box, paths["fig_dir"] / "nh_vs_tm.png")
    save_figure(fig_leak, paths["fig_dir"] / "leakiness_vs_delta_p.png")
    save_figure(fig_parity, paths["fig_dir"] / "vienna_nupack_parity.png")
    save_figure(fig_nov, paths["fig_dir"] / "novelty_vs_nh.png")
    conc = engine_concordance(panel)
    fragile = flag_software_fragile(panel)
    n_fragile = int(fragile.sum())
    tm_rs = conc.get("Tm", {})
    mo.vstack(
        [
            mo.md(
                f"Panel-wide Vienna–NUPACK $T_m$: $r_s$={tm_rs.get('r_s')}, "
                f"MAE={tm_rs.get('mae')}, n={tm_rs.get('n')}. "
                f"Software-fragile rows: **{n_fragile}**."
            ),
            fig_box,
            fig_leak,
            fig_parity,
            fig_nov,
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## §5 Two-tier gate checklist

    > Do natural thermoswitches and EVA passers clear a *plausible* melt (Tier 1),
    > or only the strict Neupert heat-inducible RBS-unmasking spec (Tier 2)?

    <br>
    **Tier 1 — Biological plausibility**
    <br>
    <br>
    - $n_{\mathrm{H}} > 1.2$
    <br>
    - $T_{\mathrm{m}} \in [35, 55]^\circ\mathrm{C}$
    <br>
    - $\Delta\theta \ge 0.20$
    <br>
    - $P_{\mathrm{open}}^{37^\circ\mathrm{C}} \le 0.35$

    <br>
    **Purpose:**
    <br>
    Non-zero survival baseline for natural Rfam; ask whether EVA samples
    biologically viable folds.

    <br><br>
    **Tier 2 — Neupert 2008 / synbio spec**
    <br>
    <br>
    - $n_{\mathrm{H}} > 1.5$
    <br>
    - $T_{\mathrm{m}} \in [42, 45]^\circ\mathrm{C}$
    <br>
    - $\Delta\theta \ge 0.40$ (ideal $\rightarrow 0.50$)
    <br>
    - $P_{\mathrm{open}}^{37^\circ\mathrm{C}} \le 0.20$

    <br>
    **Purpose:**
    <br>
    Turnkey *E. coli* heat-shock controllers without secondary rational tuning.

    <br><br>
    **Literature roles**
    <br>
    <br>
    **Neupert et al. (2008)** — primary Tier 2 biophysical spec:
    <br>
    - heat-*inducible* RNA thermometers,
    <br>
    - ON at high $T$ via Shine–Dalgarno unmasking,
    <br>
    - matches EVA task, Rfam thermoswitches, FourU archetype.

    <br>
    **Hoynes-O’Connor et al. (2015)** — methodological benchmark:
    <br>
    - heat-*repressible* RNase E design class (not Tier 2 math),
    <br>
    - dual controls + orthogonality + mechanism rescue layout,
    <br>
    - maps to FourU overlay + §12 disrupt/rescue causality audits.

    <br>
    ```text
    Hoynes-O'Connor experimental layout (method template)
    1. Transcriptional scanning (pTet × aTc burden window)
    2. Steady-state ON/OFF + No-ARC / always-ON dual controls
    3. Orthogonality (Mg²⁺ starvation, acid stress at fixed T)
    4. Mechanism proof (rne131 collapse + rne rescue / RT-qPCR)
    5. Modular circuit logic (2-/3-input AND gates)
    ```
    """)
    return


@app.cell
def _(checklist_by_cohort, mo, panel, paths, pd, write_checklist_summary):
    checklist = checklist_by_cohort(panel)
    if not checklist.empty:
        write_checklist_summary(checklist, paths["checklist_json"])

    tier1 = checklist[
        [
            c
            for c in (
                "cohort",
                "n",
                "t1_passed_nh",
                "t1_passed_tm",
                "t1_passed_amp",
                "t1_passed_base",
                "t1_passed_all_four",
                "t1_frac_all_four",
            )
            if c in checklist.columns
        ]
    ].rename(
        columns={
            "cohort": "Cohort",
            "n": "N",
            "t1_passed_nh": "T1 n_H",
            "t1_passed_tm": "T1 Tm",
            "t1_passed_amp": "T1 Δθ",
            "t1_passed_base": "T1 base",
            "t1_passed_all_four": "T1 all 4",
            "t1_frac_all_four": "T1 frac",
        }
    )
    tier2 = checklist[
        [
            c
            for c in (
                "cohort",
                "n",
                "t2_passed_nh",
                "t2_passed_tm",
                "t2_passed_amp",
                "t2_passed_base",
                "t2_passed_all_four",
                "t2_frac_all_four",
                "frac_target_box",
            )
            if c in checklist.columns
        ]
    ].rename(
        columns={
            "cohort": "Cohort",
            "n": "N",
            "t2_passed_nh": "T2 n_H",
            "t2_passed_tm": "T2 Tm",
            "t2_passed_amp": "T2 Δθ",
            "t2_passed_base": "T2 base",
            "t2_passed_all_four": "T2 all 4",
            "t2_frac_all_four": "T2 frac",
            "frac_target_box": "T2 box (Tm∧n_H)",
        }
    )
    mo.vstack(
        [
            mo.md("### Tier 1 — Biological plausibility"),
            mo.ui.table(tier1.round(4) if not tier1.empty else pd.DataFrame()),
            mo.md("### Tier 2 — Neupert 2008 / synbio spec"),
            mo.ui.table(tier2.round(4) if not tier2.empty else pd.DataFrame()),
        ]
    )
    return


@app.cell(hide_code=True)
def _(SCORE_WEIGHTS, mo):
    mo.md(rf"""
    ## §6 Top-10 experimental leads (Tier-2-aligned ranking)

    > After capping $n_H$ and severely penalizing weak stroke / leak — which EVA
    > passers remain nominable for wet-lab follow-up?

    <br>
    $$\mathrm{{Score}}
    = w_1 \cdot \min(n_{{\mathrm{{H}}}},\, 3.0)
    + w_2 \cdot \lvert\Delta\theta\rvert
    - w_3 \cdot \lvert T_{{\mathrm{{m}}}} - 43.5\rvert
    - w_4 \cdot P_{{\mathrm{{open}}}}^{{37^\circ\mathrm{{C}}}}
    - w_5 \cdot \lvert \Delta T_{{\mathrm{{m}}}}^{{\mathrm{{Vienna-NP}}}}\rvert
    - \mathrm{{Penalty}}_{{\mathrm{{stroke}}}}$$

    <br>
    $\mathrm{{Penalty}}_{{\mathrm{{stroke}}}}$
    $= w_{{\mathrm{{stroke}}}} \cdot \max(0,\, {SCORE_WEIGHTS["amp_floor"]} - \lvert\Delta\theta\rvert)$
    when stroke is below the floor (drops near-zero-amplitude fits).

    <br>
    **Direct adjustments:**
    <br>
    - clip $n_H$ at {SCORE_WEIGHTS["n_h_cap"]} (no $n_H=20$ advantage),
    <br>
    - reward $w_2\cdot\lvert\Delta\theta\rvert$ ($w_2$={SCORE_WEIGHTS["w_amp"]}),
    <br>
    - severe stroke penalty $w_{{\mathrm{{stroke}}}}$={SCORE_WEIGHTS["w_stroke_penalty"]}
      if $\lvert\Delta\theta\rvert < {SCORE_WEIGHTS["amp_floor"]}$,
    <br>
    - basal weight $w_4$={SCORE_WEIGHTS["w_leak"]}
      (favor $P_{{\mathrm{{open}}}}^{{37}} \le 0.20$).

    <br>
    Other weights: $w_1$={SCORE_WEIGHTS["w_hill"]}, $w_3$={SCORE_WEIGHTS["w_tm"]},
    $w_5$={SCORE_WEIGHTS["w_engine"]}.

    <br>
    Restriction screens: EcoRI, BamHI, XhoI.
    <br>
    After re-rank: run §7 heatmaps, §10 MDS, §12 disrupt/rescue on the **top 3**.
    """)
    return


@app.cell
def _(SCORE_WEIGHTS, eva_fitted, export_leads, mo, paths, pd, rank_leads):
    leads = rank_leads(eva_fitted, n=10)
    lead_path = None
    if not leads.empty and leads["sequence"].notna().any():
        lead_path = export_leads(
            leads, fasta_path=paths["leads_fasta"], fig_dir=paths["fig_dir"] / "leads"
        )
    show_cols = [
        c
        for c in (
            "record_id",
            "cohort",
            "composite_score",
            "viennarna_hill_coeff",
            "viennarna_Tm",
            "viennarna_amplitude",
            "viennarna_delta_P_RBS",
            "viennarna_P_open_RBS_37",
            "viennarna_mfe_zscore",
            "E_Rfam",
            "software_fragile",
        )
        if c in leads.columns
    ]
    mo.vstack(
        [
            mo.md(
                f"Exported `{lead_path}`."
                if lead_path
                else "No EVA sequences to export (FASTA missing or empty)."
            ),
            mo.md(f"Score weights: `{SCORE_WEIGHTS}`"),
            mo.ui.table(
                leads[show_cols].round(4) if not leads.empty else pd.DataFrame()
            ),
        ]
    )
    return (leads,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## §7 Positional unpairing heatmaps (local vs global melt)

    **What:** temperature on $x$, sequence position on $y$, color = unpaired chance.

    **Why:** a real switch should unzip mainly at the Shine–Dalgarno / RBS hairpin.
    If the *whole* transcript lights up, it is falling apart — not regulating.

    **Analogy:** a zipper that only opens near the pull-tab (RBS) vs a coat that
    unravels from every seam at once.

    Toggle **Compute positional unpairing heatmaps** above (top-3 scored leads).
    """)
    return


@app.cell
def _(
    leads,
    mo,
    paths,
    plot_positional_unpairing_heatmap,
    positional_unpairing_matrix,
    run_heatmaps,
    save_figure,
    sd_span_for_sequence,
    sweep_temps,
):
    heatmap_figs = []
    if run_heatmaps.value and not leads.empty:
        _temps_hm = sweep_temps()
        for _, _row_hm in leads.head(3).iterrows():
            _seq_hm = str(_row_hm.get("sequence") or "")
            if not _seq_hm:
                continue
            matrix = positional_unpairing_matrix(_seq_hm, _temps_hm)
            fig_h = plot_positional_unpairing_heatmap(
                matrix,
                _temps_hm,
                title=f"{_row_hm['record_id']} positional melt",
                sd_span=sd_span_for_sequence(_seq_hm),
            )
            save_figure(fig_h, paths["fig_dir"] / f"heatmap_{_row_hm['record_id']}.png")
            heatmap_figs.append(fig_h)
    note = (
        f"Rendered {len(heatmap_figs)} lead heatmap(s)."
        if heatmap_figs
        else "Heatmaps off or no lead sequences — flip the switch to compute."
    )
    mo.vstack([mo.md(note), *heatmap_figs])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## §8 Operational induction ratio (Fold-Change ON/OFF)

    $$\mathrm{Fold\text{-}Change}
    = P_{\mathrm{open}}(42^\circ\mathrm{C}) \,/\, P_{\mathrm{open}}(37^\circ\mathrm{C})$$

    **What:** how many times more open the RBS is at heat-shock vs resting temp.

    **Why:** $\Delta P_{\mathrm{RBS}}$ is an absolute gap. Fold-change is the biological
    signal-to-noise a ribosome would "feel."

    **Looking for:** RefSeq near 1 (little induction); Rfam and EVA clearly above 1.
    Uses sweeps when present, else Hill reconstruction, else a 55/37 proxy.
    """)
    return


@app.cell
def _(
    add_induction_ratio,
    mo,
    panel,
    paths,
    plot_induction_ratio_violins,
    save_figure,
    sweeps,
):
    panel_fc = add_induction_ratio(panel, sweeps=sweeps)
    fig_fc = plot_induction_ratio_violins(panel_fc)
    save_figure(fig_fc, paths["fig_dir"] / "induction_fold_change.png")
    src_counts = (
        panel_fc["fold_change_source"].value_counts().to_dict()
        if "fold_change_source" in panel_fc.columns
        else {}
    )
    mo.vstack(
        [
            mo.md(f"Fold-change sources: `{src_counts}`"),
            fig_fc,
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## §9 First-derivative sharpness ($\mathrm{d}\theta/\mathrm{d}T$)

    **What:** slope of the melting curve vs temperature — cooperativity as a peak.

    **Why:** high $n_{\mathrm{H}}$ should make a tall, narrow bump near $T_m$. Soft melts
    look like low, wide hills.

    **Looking for:** peaks inside the shaded 42–45 °C band with small FWHM.
    Flat / smeared lines are non-switching controls.
    """)
    return


@app.cell
def _(
    derivative_profiles,
    mo,
    np,
    panel,
    paths,
    plot_derivative_curves,
    save_figure,
    sweep_temps,
):
    temps_d = np.asarray(sweep_temps(), dtype=float)
    deriv = derivative_profiles(panel, temps_d, max_per_cohort=35)
    fig_d = plot_derivative_curves(deriv)
    save_figure(fig_d, paths["fig_dir"] / "dtheta_dT.png")
    fwhm_summary = (
        deriv.groupby("cohort")["fwhm_C"].median().rename("median_FWHM_C").reset_index()
        if not deriv.empty
        else None
    )
    _blocks_d = [fig_d]
    if fwhm_summary is not None and not fwhm_summary.empty:
        _blocks_d.extend(
            [
                mo.md("### Median FWHM of $\\mathrm{d}\\theta/\\mathrm{d}T$ peak (°C)"),
                mo.ui.table(fwhm_summary.round(3)),
            ]
        )
    mo.vstack(_blocks_d)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## §10 Structural diversity embedding (mode-collapse check)

    **What:** fold each EVA sequence to its MFE structure, measure Tree Edit Distance
    between folds, then place them on a 2D MDS map.

    **Why:** a generator can spit out many sequences that are tiny edits of one hairpin.
    Sequence novelty ≠ fold novelty.

    **Looking for:** several islands (diverse scaffolds). One tight blob = mode collapse.

    Toggle **Embed EVA MFE structures** above.
    """)
    return


@app.cell
def _(
    embed_structure_manifold,
    eva_fitted,
    mo,
    paths,
    plot_structure_manifold,
    run_structure_embed,
    save_figure,
    structure_distance_matrix,
):
    fig_struct = None
    embed_note = (
        "Structure embedding off — flip the switch to run MDS on EVA MFE folds."
    )
    if run_structure_embed.value and not eva_fitted.empty:
        eva_seq = eva_fitted.dropna(subset=["sequence"]).head(80)
        dist, _structs = structure_distance_matrix(
            eva_seq["sequence"].astype(str).tolist()
        )
        coords, method_name = embed_structure_manifold(dist, method="MDS")
        fig_struct = plot_structure_manifold(
            coords, eva_seq["cohort"].tolist(), method_name=method_name
        )
        save_figure(fig_struct, paths["fig_dir"] / "structure_manifold.png")
        embed_note = (
            f"Embedded {len(eva_seq)} EVA folds with **{method_name}** "
            f"(pairwise tree-edit / BP distances)."
        )
    mo.vstack([mo.md(embed_note), *([fig_struct] if fig_struct is not None else [])])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## §11 Tier-2 (Neupert) survival funnel

    **What:** consecutive attrition under Tier 2 gates
    ($n_H \to T_m \to \Delta\theta \to$ baseline).

    **Why:** Tier 1 can look healthy while Tier 2 yield is still zero.
    The funnel is the honest KPI for synbio-ready heat-inducible switches.

    **Looking for:** Rfam / EVA may retain Tier 1 survivors (§5), but Tier 2
    often ends at ~0 — matching the Neupert-strict finding in the summary.
    """)
    return


@app.cell
def _(
    gate_survival_funnel,
    mo,
    panel,
    paths,
    plot_gate_survival_funnel,
    save_figure,
):
    funnel = gate_survival_funnel(panel)
    fig_funnel = plot_gate_survival_funnel(funnel)
    save_figure(fig_funnel, paths["fig_dir"] / "gate_survival_funnel.png")
    mo.vstack(
        [
            fig_funnel,
            mo.md("### Funnel table"),
            mo.ui.table(funnel.round(4) if not funnel.empty else funnel),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## §12 In silico mutational disruption & rescue

    **What:** on the top-3 leads, break one RBS-stem base pair (mismatch), then
    fix the partner base so pairing can return (compensatory rescue).

    **Why:** proves the melt depends on that stem pair, not just overall %GC.

    **Looking for:** disrupt curve leaks or loses snap; rescue moves back toward
    wild-type. If all three curves look the same, the phenotype is composition-driven.

    Toggle **Run in-silico disrupt/rescue** above (Vienna temperature sweeps).
    """)
    return


@app.cell
def _(
    characterize_mutation_trio,
    leads,
    mo,
    paths,
    plot_mutation_rescue_curves,
    run_mutations,
    save_figure,
):
    mut_curves: dict = {}
    fig_mut = None
    mut_note = "Mutation rescue off — flip the switch to sweep top-3 leads."
    if run_mutations.value and not leads.empty:
        for _, _row_mut in leads.head(3).iterrows():
            _seq_mut = str(_row_mut.get("sequence") or "")
            rid = str(_row_mut.get("record_id") or "lead")
            if not _seq_mut:
                continue
            mut_curves[rid] = characterize_mutation_trio(_seq_mut)
        fig_mut = plot_mutation_rescue_curves(mut_curves)
        save_figure(fig_mut, paths["fig_dir"] / "mutation_rescue.png")
        mut_note = f"Swept disrupt/rescue for {len(mut_curves)} lead(s)."
    _blocks_mut = [mo.md(mut_note)]
    if fig_mut is not None:
        _blocks_mut.append(fig_mut)
    mo.vstack(_blocks_mut)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## §13 Proof-of-Concept: Downstream Rational Tuning (Lead 315)

    **Hypothesis:** Introducing targeted G–C clamps to the non-RBS scaffold stems of
    `eva_sample_315` will suppress global background melting at 42–45 °C while
    preserving Shine–Dalgarno unpairing.

    **What:** pick the 3 farthest non-SD A–U / U–A MFE pairs, mutate them to G–C /
    C–G, then re-run `positional_unpairing_matrix` and compare Before vs After.

    **Why:** App 8 showed lead 315 melts globally rather than as a modular RBS
    cassette. A one-sequence clamp test asks whether that fraying is a local
    scaffold weakness (fixable by hand) or a distributed ensemble problem.

    **Looking for:** Scenario 1 (scaffold dark, SD stroke ↑), Scenario 2 (SD
    locked / negative design trap), or Scenario 3 (scaffold still frays).
    """)
    return


@app.cell
def _(
    eva,
    mo,
    paths,
    plot_before_after_positional_heatmaps,
    run_scaffold_gc_clamp_poc,
    save_figure,
):
    """Manual G–C clamp rescue on eva_sample_315 with before/after heatmap."""
    import json

    poc_note = "eva_sample_315 not found in EVA passers — skip §13 PoC."
    poc_blocks: list = []
    row_315 = eva.loc[eva["record_id"].astype(str) == "eva_sample_315"]
    if not row_315.empty:
        seq_315 = str(row_315.iloc[0].get("sequence") or "")
        poc_315 = run_scaffold_gc_clamp_poc(
            seq_315,
            record_id="eva_sample_315",
            n_pairs=3,
        )
        fig_poc = plot_before_after_positional_heatmaps(poc_315)
        save_figure(fig_poc, paths["fig_dir"] / "poc_315_before_after_heatmap.png")
        m_wt = poc_315["wildtype_metrics"]
        m_mut = poc_315["clamped_metrics"]
        scenario = int(poc_315["scenario"])
        scenario_name = {
            1: "Ideal Modular Rescue",
            2: "Negative Design Trap",
            3: "Partial / Ineffective Rescue",
        }.get(scenario, f"Scenario {scenario}")
        edits = poc_315.get("edits") or []
        edit_txt = ", ".join(
            f"nt{e['i'] + 1}–{e['j'] + 1}: {e['from']}→{e['to']}" for e in edits
        )
        # Persist metrics without huge matrices for reproducibility.
        slim = {
            k: v for k, v in poc_315.items() if k not in {"mat_wildtype", "mat_clamped"}
        }
        (paths["cache_dir"] / "poc_315_gc_clamp.json").write_text(
            json.dumps(slim, indent=2)
        )
        poc_note = (
            f"**Outcome: Scenario {scenario} — {scenario_name}.** "
            f"Clamped {len(edits)} distant scaffold pair(s) ({edit_txt}). "
            f"Global $P_{{\\mathrm{{unpaired}}}}$ at 43 °C: "
            f"{m_wt.get('global_43', float('nan')):.2f} → {m_mut.get('global_43', float('nan')):.2f}; "
            f"SD stroke $\\Delta\\theta$(43−37): "
            f"{m_wt.get('stroke_sd_43_37', float('nan')):.2f} → "
            f"{m_mut.get('stroke_sd_43_37', float('nan')):.2f}; "
            f"SD $P_{{\\mathrm{{open}}}}$ at 37 °C: "
            f"{m_wt.get('sd_37', float('nan')):.2f} → {m_mut.get('sd_37', float('nan')):.2f}."
        )
        poc_blocks = [
            mo.md("### Case study box — Before vs After"),
            fig_poc,
            mo.md(poc_note),
            mo.md(
                r"""
                **How to read the heatmaps.** Magma bright = unpaired. Cyan band =
                Shine–Dalgarno (nt ~400–405). White vertical band = 42–45 °C.
                Ideal rescue would keep the scaffold dark while only the cyan
                band brightens at heat shock.
                """
            ),
        ]
    else:
        poc_blocks = [mo.md(poc_note)]
    mo.vstack(poc_blocks)
    return


if __name__ == "__main__":
    app.run()
