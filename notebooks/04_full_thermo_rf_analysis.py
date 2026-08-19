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
    # Full Thermodynamic RF Pipeline — Research Analysis

    End-to-end analysis of the Mac M3 production run:

    1. **Training corpus** — 2,395 Rfam balanced thermoswitch / non-switch sequences with ViennaRNA + NUPACK physics features
    2. **Generative corpus** — 10,000 de novo RNA sequences from GenerRNA (RunPod)
    3. **Classifier** — Random Forest trained on 15 fused physics features, applied to de novo sequences

    This notebook addresses questions a methods/results section would typically cover: dataset composition, model discrimination, feature attribution, inter-engine agreement, train→de novo domain shift, prediction calibration, biophysical plausibility of hits, and candidate prioritisation.
    """)
    return


@app.cell
def _():
    # '%matplotlib inline' command supported automatically in marimo

    import json
    import sys
    from pathlib import Path

    import joblib
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from scipy import stats
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (
        ConfusionMatrixDisplay,
        RocCurveDisplay,
        classification_report,
        roc_auc_score,
    )
    from sklearn.model_selection import (
        StratifiedKFold,
        cross_validate,
        cross_val_predict,
    )

    PROJECT_ROOT = Path.cwd().resolve()
    if not (PROJECT_ROOT / "src").exists():
        PROJECT_ROOT = PROJECT_ROOT.parent
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

    DATA = PROJECT_ROOT / "data" / "processed"
    FIG_DIR = DATA / "figures" / "full_analysis"
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid", context="notebook", palette="colorblind")
    plt.rcParams.update({"figure.dpi": 120, "savefig.bbox": "tight"})

    TARGET_TM_LO, TARGET_TM_HI = (
        37.0,
        55.0,
    )  # °C design window for bacterial ON switches
    return (
        ConfusionMatrixDisplay,
        DATA,
        FIG_DIR,
        RandomForestClassifier,
        RocCurveDisplay,
        StratifiedKFold,
        TARGET_TM_HI,
        TARGET_TM_LO,
        classification_report,
        cross_val_predict,
        cross_validate,
        joblib,
        json,
        np,
        pd,
        plt,
        roc_auc_score,
        sns,
        stats,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Data loading and pipeline metadata
    """)
    return


@app.cell
def _(DATA, joblib, pd):
    train = pd.read_csv(DATA / "fused_features.csv")
    denovo = pd.read_csv(DATA / "denovo_fused_features.csv")
    pred = pd.read_csv(DATA / "denovo_predictions.csv")
    payload = joblib.load(DATA / "models" / "rf_thermoswitch.joblib")
    model = payload["model"]
    feature_cols = payload["feature_columns"]

    merged = pred.merge(denovo, on="record_id", how="inner", validate="one_to_one")
    assert len(merged) == len(pred), "Prediction / feature row mismatch"

    train_clean = train.dropna(subset=["label"] + feature_cols)
    denovo_clean = denovo.dropna(subset=feature_cols)

    pipeline_meta = {
        "training_rows": len(train_clean),
        "training_pos": int((train_clean["label"] == 1).sum()),
        "training_neg": int((train_clean["label"] == 0).sum()),
        "denovo_rows": len(denovo),
        "predictions": len(pred),
        "predicted_pos": int((pred["predicted_label"] == 1).sum()),
        "predicted_neg": int((pred["predicted_label"] == 0).sum()),
        "n_features": len(feature_cols),
        "temp_grid": "37,41,45,49,53,55 °C",
        "salt": "0.05 M Na+, 0.0 M Mg2+ (NUPACK rna06)",
    }
    pd.Series(pipeline_meta, name="value").to_frame()
    return (
        denovo_clean,
        feature_cols,
        merged,
        model,
        pipeline_meta,
        train_clean,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Training corpus characterisation

    **Research question:** Is the labelled corpus balanced and structurally diverse enough to support supervised learning?
    """)
    return


@app.cell
def _(FIG_DIR, TARGET_TM_HI, TARGET_TM_LO, plt, train_clean):
    _fig, _axes = plt.subplots(1, 3, figsize=(13, 3.8))
    train_clean["label"].map(
        {0: "non-switch", 1: "thermoswitch"}
    ).value_counts().plot.bar(ax=_axes[0], color=["#4C72B0", "#DD8452"])
    _axes[0].set_title("Class balance (training)")
    _axes[0].set_ylabel("count")
    _axes[0].tick_params(axis="x", rotation=0)
    for _label, color, _name in [
        (0, "#4C72B0", "non-switch"),
        (1, "#DD8452", "thermoswitch"),
    ]:
        _subset = train_clean.loc[train_clean["label"] == _label, "seq_length"]
        _axes[1].hist(
            _subset, bins=30, alpha=0.55, label=_name, color=color, density=True
        )
    _axes[1].set_xlabel("sequence length (nt)")
    _axes[1].set_title("Length distribution by class")
    _axes[1].legend()
    for _label, color, _name in [
        (0, "#4C72B0", "non-switch"),
        (1, "#DD8452", "thermoswitch"),
    ]:
        _subset = train_clean.loc[train_clean["label"] == _label, "nupack_Tm"]
        _axes[2].hist(
            _subset, bins=30, alpha=0.55, label=_name, color=color, density=True
        )
    _axes[2].axvspan(
        TARGET_TM_LO, TARGET_TM_HI, color="green", alpha=0.12, label="design window"
    )
    _axes[2].set_xlabel("NUPACK Tm (°C)")
    _axes[2].set_title("Melting temperature by class")
    _axes[2].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "01_training_corpus.png")
    plt.show()
    train_clean.groupby("label")[
        ["seq_length", "nupack_Tm", "nupack_amplitude", "nupack_max_stem_length"]
    ].agg(["mean", "std"]).round(2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Classifier performance (5-fold stratified CV on Rfam)

    **Research question:** How well does a physics-only Random Forest separate known thermoswitches from controls on held-out folds?

    We re-fit CV models with the same hyperparameters as production (`n_estimators=200`, `random_state=42`) to report unbiased metrics. The deployed model is trained on all 2,395 rows.
    """)
    return


@app.cell
def _(
    ConfusionMatrixDisplay,
    FIG_DIR,
    RandomForestClassifier,
    RocCurveDisplay,
    StratifiedKFold,
    classification_report,
    cross_val_predict,
    cross_validate,
    feature_cols,
    pd,
    plt,
    roc_auc_score,
    train_clean,
):
    X = train_clean[feature_cols]
    y = train_clean["label"].astype(int)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rf_cv = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    cv_scores = cross_validate(
        rf_cv,
        X,
        y,
        cv=cv,
        scoring=["accuracy", "precision", "recall", "f1", "roc_auc"],
        return_train_score=False,
    )
    cv_summary = pd.DataFrame(
        {
            metric.replace(
                "test_", ""
            ): f"{cv_scores[metric].mean():.3f} ± {cv_scores[metric].std():.3f}"
            for metric in cv_scores
            if metric.startswith("test_")
        },
        index=["5-fold CV"],
    ).T
    cv_summary.columns = ["mean ± std"]
    cv_summary
    y_proba_cv = cross_val_predict(rf_cv, X, y, cv=cv, method="predict_proba")[:, 1]
    y_pred_cv = (y_proba_cv >= 0.5).astype(int)
    _fig, _axes = plt.subplots(1, 2, figsize=(10, 4))
    RocCurveDisplay.from_predictions(
        y, y_proba_cv, ax=_axes[0], name=f"AUC={roc_auc_score(y, y_proba_cv):.3f}"
    )
    _axes[0].set_title("ROC — 5-fold out-of-fold predictions")
    ConfusionMatrixDisplay.from_predictions(y, y_pred_cv, ax=_axes[1], cmap="Blues")
    _axes[1].set_title("Confusion matrix (threshold=0.5)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "02_cv_performance.png")
    plt.show()
    print(
        classification_report(y, y_pred_cv, target_names=["non-switch", "thermoswitch"])
    )
    return (cv_scores,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Feature importance and univariate class separation

    **Research question:** Which biophysical descriptors drive discrimination — MFE, stem/loop geometry, melting-curve shape, or GC content?
    """)
    return


@app.cell
def _(FIG_DIR, feature_cols, model, np, pd, plt, train_clean):
    importance = pd.Series(model.feature_importances_, index=feature_cols).sort_values(
        ascending=True
    )
    _fig, _ax = plt.subplots(figsize=(7, 5))
    importance.plot.barh(ax=_ax, color="#4C72B0")
    _ax.set_xlabel("Gini importance (production RF)")
    _ax.set_title("Random Forest feature importance")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "03_feature_importance.png")
    plt.show()

    def cohens_d(a: pd.Series, b: pd.Series) -> float:
        """Return Cohen's d effect size for two numeric samples."""
        return (a.mean() - b.mean()) / np.sqrt(
            (a.std(ddof=1) ** 2 + b.std(ddof=1) ** 2) / 2
        )

    pos = train_clean[train_clean["label"] == 1]
    neg = train_clean[train_clean["label"] == 0]
    # Cohen's d effect size per feature (pos vs neg in training)
    effect = pd.Series(
        {col: cohens_d(pos[col], neg[col]) for col in feature_cols}
    ).sort_values(key=abs, ascending=False)
    pd.DataFrame(
        {"importance": importance.sort_values(ascending=False), "cohens_d": effect}
    ).round(3)
    return (importance,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. ViennaRNA vs NUPACK concordance

    **Research question:** Do the two physics engines agree on melting temperature and structural proxies, and does agreement differ between classes?
    """)
    return


@app.cell
def _(FIG_DIR, denovo_clean, pd, plt, stats, train_clean):
    pairs = [
        ("nupack_Tm", "viennarna_Tm", "Tm (°C)"),
        ("nupack_MFE", "viennarna_MFE", "MFE (kcal/mol)"),
        ("nupack_gc_content", "viennarna_gc_content", "GC fraction"),
    ]
    _fig, _axes = plt.subplots(2, 3, figsize=(12, 7))
    cor_rows = []
    for col, (xcol, ycol, _label) in enumerate(pairs):
        for row, (_subset, _title) in enumerate(
            [(train_clean, "Rfam training"), (denovo_clean, "De novo 10k")]
        ):
            _ax = _axes[row, col]
            _r, _p = stats.pearsonr(_subset[xcol], _subset[ycol])
            cor_rows.append({"dataset": _title, "pair": _label, "r": _r, "p": _p})
            _ax.scatter(_subset[xcol], _subset[ycol], s=8, alpha=0.25)
            lims = [
                min(_subset[xcol].min(), _subset[ycol].min()),
                max(_subset[xcol].max(), _subset[ycol].max()),
            ]
            _ax.plot(lims, lims, "k--", lw=1, alpha=0.5)
            _ax.set_xlabel(f"NUPACK {_label}")
            _ax.set_ylabel(f"ViennaRNA {_label}")
            _ax.set_title(f"{_title}\nr={_r:.2f}")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "04_engine_concordance.png")
    plt.show()
    pd.DataFrame(cor_rows).round(4)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Domain shift: Rfam training → GenerRNA de novo

    **Research question:** Does the generative model produce sequences whose physics feature distributions differ from natural Rfam switches (covariate shift)?
    """)
    return


@app.cell
def _(FIG_DIR, denovo_clean, pd, plt, sns, stats, train_clean):
    shift_features = [
        "seq_length",
        "nupack_Tm",
        "nupack_amplitude",
        "nupack_max_stem_length",
        "nupack_gc_content",
        "viennarna_mean_unpaired_prob",
    ]
    ks_rows = []
    _fig, _axes = plt.subplots(2, 3, figsize=(12, 6))
    _axes = _axes.ravel()
    for _ax, _feat in zip(_axes, shift_features):
        a = train_clean[_feat].dropna()
        b = denovo_clean[_feat].dropna()
        stat, _p = stats.ks_2samp(a, b)
        ks_rows.append({"feature": _feat, "ks_stat": stat, "p_value": _p})
        sns.kdeplot(a, ax=_ax, label="Rfam train", fill=True, alpha=0.35)
        sns.kdeplot(b, ax=_ax, label="de novo", fill=True, alpha=0.35)
        _ax.set_title(f"{_feat}\nKS={stat:.3f}")
        _ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "05_domain_shift.png")
    plt.show()
    pd.DataFrame(ks_rows).sort_values("ks_stat", ascending=False).round(4)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. De novo prediction landscape

    **Research question:** What fraction of AI-generated sequences are classified as thermoswitches, and how confident are those calls?
    """)
    return


@app.cell
def _(FIG_DIR, merged, plt):
    _fig, _axes = plt.subplots(1, 2, figsize=(10, 4))
    _axes[0].hist(merged["prob_positive"], bins=40, color="#4C72B0", edgecolor="white")
    _axes[0].axvline(0.5, color="red", ls="--", lw=1, label="decision threshold")
    _axes[0].set_xlabel("P(thermoswitch)")
    _axes[0].set_ylabel("count")
    _axes[0].set_title("Score distribution — 9,999 de novo sequences")
    _axes[0].legend()
    merged["predicted_label"].map(
        {0: "predicted non-switch", 1: "predicted thermoswitch"}
    ).value_counts().plot.bar(ax=_axes[1], color=["#4C72B0", "#DD8452"])
    _axes[1].set_title("Hard labels at threshold 0.5")
    _axes[1].tick_params(axis="x", rotation=15)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "06_prediction_landscape.png")
    plt.show()
    hit_rate = merged["predicted_label"].mean()
    high_conf = (merged["prob_positive"] >= 0.9).sum()
    print(
        f"Hit rate (label=1): {hit_rate:.1%}  "
        f"({merged['predicted_label'].sum():,} / {len(merged):,})"
    )
    print(
        f"High-confidence hits (P≥0.9): {high_conf:,} ({high_conf / len(merged):.1%})"
    )
    return high_conf, hit_rate


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. Biophysical plausibility of predicted positives

    **Research question:** Do predicted thermoswitches exhibit switch-like melting behaviour (sigmoidal amplitude, Tm near the design window, adequate stem/loop geometry)?
    """)
    return


@app.cell
def _(FIG_DIR, TARGET_TM_HI, TARGET_TM_LO, merged, pd, plt, sns, train_clean):
    pred_pos = merged[merged["predicted_label"] == 1]
    pred_neg = merged[merged["predicted_label"] == 0]
    train_pos = train_clean[train_clean["label"] == 1]
    bio_metrics = []
    for _name, df in [
        ("Rfam positives (reference)", train_pos),
        ("Predicted de novo positives", pred_pos),
        ("Predicted de novo negatives", pred_neg),
    ]:
        in_window = (
            (df["nupack_Tm"] >= TARGET_TM_LO) & (df["nupack_Tm"] <= TARGET_TM_HI)
        ).mean()
        bio_metrics.append(
            {
                "cohort": _name,
                "n": len(df),
                "nupack_Tm_mean": df["nupack_Tm"].mean(),
                "nupack_amplitude_mean": df["nupack_amplitude"].mean(),
                "nupack_hill_mean": df["nupack_hill_coeff"].mean(),
                "stem_len_mean": df["nupack_max_stem_length"].mean(),
                "loop_len_mean": df["nupack_max_loop_length"].mean(),
                "frac_Tm_in_37_55C": in_window,
            }
        )
    bio_df = pd.DataFrame(bio_metrics).round(3)
    bio_df
    _fig, _axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for _ax, _feat, _title in zip(
        _axes,
        ["nupack_Tm", "nupack_amplitude", "nupack_max_stem_length"],
        ["NUPACK Tm", "NUPACK amplitude", "Max stem length"],
    ):
        sns.kdeplot(train_pos[_feat], ax=_ax, label="Rfam +", fill=True, alpha=0.3)
        sns.kdeplot(
            pred_pos[_feat], ax=_ax, label="de novo pred +", fill=True, alpha=0.3
        )
        sns.kdeplot(
            pred_neg[_feat], ax=_ax, label="de novo pred −", fill=True, alpha=0.3
        )
        if _feat == "nupack_Tm":
            _ax.axvspan(TARGET_TM_LO, TARGET_TM_HI, color="green", alpha=0.1)
        _ax.set_title(_title)
        _ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "07_biophysical_plausibility.png")
    plt.show()
    return (bio_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9. Score vs biophysics — does the RF track known switch signatures?

    **Research question:** Are higher RF scores associated with stronger melting transitions and more switch-like Tm values on de novo sequences?
    """)
    return


@app.cell
def _(FIG_DIR, merged, plt, stats):
    scatter_feats = [
        "nupack_Tm",
        "nupack_amplitude",
        "nupack_MFE",
        "nupack_max_stem_length",
    ]
    _fig, _axes = plt.subplots(2, 2, figsize=(9, 7))
    for _ax, _feat in zip(_axes.ravel(), scatter_feats):
        sub = merged.sample(min(3000, len(merged)), random_state=42)
        sc = _ax.scatter(
            sub[_feat],
            sub["prob_positive"],
            c=sub["prob_positive"],
            cmap="viridis",
            s=10,
            alpha=0.5,
        )
        _r, _p = stats.pearsonr(merged[_feat], merged["prob_positive"])
        _ax.set_xlabel(_feat)
        _ax.set_ylabel("P(thermoswitch)")
        _ax.set_title(f"r={_r:.2f}")
    _fig.colorbar(sc, ax=_axes.ravel().tolist(), label="P(thermoswitch)", shrink=0.8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "08_score_vs_biophysics.png")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 10. Candidate prioritisation for experimental validation

    **Research question:** Which de novo sequences should be synthesised first?

    We rank by RF probability and apply biophysical filters:
    - NUPACK Tm within 37–55 °C (ON at physiological temperature)
    - NUPACK amplitude ≥ 0.15 (detectable melting transition)
    - Hill coefficient ≥ 2 (cooperative transition)
    - Vienna + NUPACK fit status = `ok`
    """)
    return


@app.cell
def _(TARGET_TM_HI, TARGET_TM_LO, merged):
    candidates = merged[
        (merged["prob_positive"] >= 0.9)
        & (merged["nupack_Tm"].between(TARGET_TM_LO, TARGET_TM_HI))
        & (merged["nupack_amplitude"] >= 0.15)
        & (merged["nupack_hill_coeff"] >= 2)
        & (merged["nupack_fit_status"] == "ok")
        & (merged["viennarna_fit_status"] == "ok")
    ].copy()

    candidates["priority_score"] = (
        candidates["prob_positive"]
        + 0.15 * candidates["nupack_amplitude"].clip(0, 1)
        + 0.05 * (candidates["nupack_hill_coeff"].clip(0, 20) / 20)
    )
    candidates = candidates.sort_values("priority_score", ascending=False)

    display_cols = [
        "record_id",
        "prob_positive",
        "priority_score",
        "seq_length",
        "nupack_Tm",
        "nupack_amplitude",
        "nupack_hill_coeff",
        "nupack_max_stem_length",
        "nupack_max_loop_length",
        "nupack_gc_content",
    ]
    print(f"Filtered candidates passing all gates: {len(candidates):,}")
    candidates[display_cols].head(20)
    return candidates, display_cols


@app.cell
def _(DATA, candidates, display_cols):
    export_path = DATA / "denovo_top_candidates.csv"
    candidates[display_cols].to_csv(export_path, index=False)
    print(f"Exported top candidates table → {export_path}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 11. Synthesis — key findings for a paper

    Run the cell below to emit a JSON summary suitable for a results paragraph or supplementary table.
    """)
    return


@app.cell
def _(
    DATA,
    bio_df,
    candidates,
    cv_scores,
    high_conf,
    hit_rate,
    importance,
    json,
    pipeline_meta,
):
    summary = {
        "methods": pipeline_meta,
        "cv_performance": {
            k.replace("test_", ""): float(cv_scores[k].mean())
            for k in cv_scores
            if k.startswith("test_")
        },
        "top_features": importance.sort_values(ascending=False)
        .head(5)
        .round(4)
        .to_dict(),
        "denovo_predictions": {
            "hit_rate_0.5": float(hit_rate),
            "high_conf_ge_0.9": int(high_conf),
            "filtered_candidates": int(len(candidates)),
        },
        "biophysical_cohorts": bio_df.to_dict(orient="records"),
        "limitations": [
            "No experimental ground truth for de novo hits — all labels are model-derived.",
            "Training labels are Rfam homology clusters, not wet-lab validated switches.",
            "Sparse temperature grid (6 points) may bias Hill/Tm fits vs full 10–80 °C prototype panel.",
            "One Rfam sequence skipped (IUPAC ambiguity); one de novo sequence has flat Vienna fit.",
            "Generative sequences may sit outside Rfam manifold (KS domain shift on several features).",
        ],
    }

    summary_path = DATA / "full_analysis_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nSaved → {summary_path}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Interpretation checklist

    | Question | Where answered |
    |---|---|
    | Is the training set balanced? | §2 |
    | Does physics-only RF discriminate known switches? | §3 (AUC ~0.95 CV) |
    | Which features matter most? | §4 (MFE dominates; GC, stem/loop secondary) |
    | Do ViennaRNA and NUPACK agree? | §5 |
    | Is there covariate shift to GenerRNA? | §6 (KS tests) |
    | What is the de novo hit rate? | §7 (~39% at 0.5 threshold) |
    | Are hits biophysically plausible? | §8–9 |
    | What to synthesise first? | §10 (`denovo_top_candidates.csv`) |

    **Suggested next experiments:** FRET or SHAPE-MaP melting curves on top-20 candidates; compare predicted Tm/amplitude to measured; use BiRNA-BERT embeddings (already in S3) for a hybrid physics+sequence model ablation.
    """)
    return


if __name__ == "__main__":
    app.run()
