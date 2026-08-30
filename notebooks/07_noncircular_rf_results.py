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
    # Non-circular RF — results (loads logged JSON)

    Written brief: [`07_noncircular_rf_model_update.md`](07_noncircular_rf_model_update.md). Architecture history: [`06_classifier_architecture_ladder.py`](06_classifier_architecture_ladder.py).

    This notebook **does not retrain**. It reads the diagnostic JSON sidecars from the 18 August 2026 non-circular RF fit.
    """)
    return


@app.cell
def _():
    # '%matplotlib inline' command supported automatically in marimo

    import json
    from pathlib import Path

    import matplotlib.pyplot as plt
    import pandas as pd
    from IPython.display import Image, display

    PROJECT_ROOT = Path.cwd().resolve()
    if not (PROJECT_ROOT / "src").exists():
        PROJECT_ROOT = PROJECT_ROOT.parent

    DATA = PROJECT_ROOT / "data" / "processed"
    FIG = PROJECT_ROOT / "notebooks" / "figures" / "07_classifier"

    diag = json.loads((DATA / "rf_noncircular_diagnostics.json").read_text())
    post = json.loads((DATA / "rf_posthoc_report.json").read_text())
    feat = json.loads((DATA / "rf_noncircular_feature_log.json").read_text())
    sidecar = json.loads(
        (DATA / "models" / "rf_thermoswitch_noncircular.json").read_text()
    )
    rus_diag_path = DATA / "rf_noncircular_diagnostics_rus.json"
    rus_diag = json.loads(rus_diag_path.read_text()) if rus_diag_path.exists() else None

    print("feature_set:", sidecar["feature_set"])
    print("trained_at:", sidecar["trained_at"])
    print(
        f"n={sidecar['n_rows']} ({sidecar['n_pos']} pos / {sidecar['n_neg']} neg), {len(sidecar['feature_columns'])} features"
    )
    print("circular excluded from X:", ", ".join(sidecar["circular_excluded"]))
    if rus_diag is not None:
        print(
            f"RUS panel GroupKFold AUC: {rus_diag['noncircular_stratified_group']['roc_auc']:.3f} "
            f"(n={rus_diag['n']})"
        )
    else:
        print("RUS diagnostics sidecar not present yet.")
    return FIG, Image, diag, display, feat, pd, plt, post, rus_diag


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## CV vs the circular 20-column RF

    Family holdout is the number that matters. Random-split 0.97 on the new matrix is k-mer family leakage, not a switch detector.
    """)
    return


@app.cell
def _(diag, display, feat, pd, rus_diag):
    cv = pd.DataFrame(
        {
            "model": [
                "Length alone",
                "Circular 20-col RF (history)",
                "Non-circular RF",
            ],
            "random_split_AUC": [
                diag["length_alone_stratified"]["roc_auc"],
                0.80,
                diag["noncircular_stratified"]["roc_auc"],
            ],
            "family_holdout_AUC": [
                None,
                0.19,
                diag["noncircular_stratified_group"]["roc_auc"],
            ],
            "note": [
                "Length shortcut removed on this panel",
                "Tm / Hill / Z / ΔP_RBS were in X (circular)",
                "92-col X; melting scalars post-hoc only",
            ],
        }
    )
    display(cv.round(3))

    if rus_diag is not None:
        cv_rus = pd.DataFrame(
            {
                "panel": ["ENN (historical)", "RUS (no ENN)"],
                "random_split_AUC": [
                    diag["noncircular_stratified"]["roc_auc"],
                    rus_diag["noncircular_stratified"]["roc_auc"],
                ],
                "family_holdout_AUC": [
                    diag["noncircular_stratified_group"]["roc_auc"],
                    rus_diag["noncircular_stratified_group"]["roc_auc"],
                ],
            }
        )
        print("ENN vs RUS non-circular RF")
        display(cv_rus.round(3))

    aug = feat["aug_missing"]
    print(
        f"Missing AUG (sentinel, not dropped): {aug['n_missing_aug_pos']} positives, "
        f"{aug['n_missing_aug_neg']} negatives (of {aug['n_pos']}/{aug['n_neg']})."
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Grouped permutation importance (in-sample)

    Full-fit AUC = 1.0, so drops describe what the overfit forest uses — not out-of-family signal. Trinucleotides dominate.
    """)
    return


@app.cell
def _(diag, display, pd, plt):
    imp = pd.DataFrame(
        [
            {
                "block": name,
                "n_features": block["n_features"],
                "mean_AUC_drop": block["mean_auc_drop"],
                "std_AUC_drop": block["std_auc_drop"],
            }
            for name, block in diag["grouped_permutation_importance"]["groups"].items()
        ]
    ).sort_values("mean_AUC_drop", ascending=False)
    display(imp.round(4))

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.barh(
        imp["block"], imp["mean_AUC_drop"], xerr=imp["std_AUC_drop"], color="#4C72B0"
    )
    ax.invert_yaxis()
    ax.set_xlabel("Grouped permutation AUC drop (in-sample)")
    ax.set_title("Non-circular RF — do not use Gini/MDI")
    fig.tight_layout()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Post-hoc (not X)

    Out-of-fold \(\hat{y}\) bins, melting gates, panel-wide Spearman (primary). High-bin \(r_s\) is omitted unless \(N \ge 25\).
    """)
    return


@app.cell
def _(display, pd, post):
    print("OOF confidence bins:", post["bins"])
    print(
        f"High-bin Spearman underpowered: {post['high_bin_spearman_underpowered']} (n_high={post['n_high']})"
    )

    gates = pd.DataFrame(
        [{"gate": k, "n_pass": v} for k, v in post["gates"].items() if v is not None]
    )
    display(gates)

    pairs = post["spearman_panel_wide_primary"]["pairs"]
    spear = pd.DataFrame(
        [
            {"pair": name, "n": s["n"], "r_s": s["r_s"], "p": s["p_value"]}
            for name, s in pairs.items()
        ]
    )
    print("Panel-wide Vienna–NUPACK Spearman (primary consensus)")
    display(spear.round(4))

    mw = pd.DataFrame(
        [
            {
                "feature": col,
                "n_high": t["n_high"],
                "n_low": t["n_low"],
                "MW_p": t["mannwhitney_p"],
                "KS_p": t["ks_p"],
            }
            for col, t in post["bin_tests_high_vs_low"].items()
        ]
    )
    print("High vs low bins (exploratory: n_high = 6)")
    display(mw.round(4))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Visual diagnostic checklist

    Snap \(n_H > 1.5\), \(T_m\) 42–45 °C, \(\Delta\theta \ge 0.50\), baseline \(P_{\mathrm{open,RBS}}(37^\circ\mathrm{C}) \le 0.20\). Intersection on this panel is zero. EVA yield is still \(Z \le -2 \land \Delta P_{\mathrm{RBS}} > 0 \land E_{\mathrm{Rfam}} > 10^{-3}\).
    """)
    return


@app.cell
def _(FIG, Image, display, pd, post):
    check = pd.DataFrame(
        [{"signature": k, "n_pass": v} for k, v in post["visual_checklist"].items()]
    )
    display(check)
    print(post["eva_hill_not_available"])

    png = FIG / "melting_visual_checklist.png"
    if png.exists():
        display(Image(filename=str(png)))
    else:
        print("Missing", png)
    return


if __name__ == "__main__":
    app.run()
