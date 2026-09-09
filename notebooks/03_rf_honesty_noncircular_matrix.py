"""03 · RF honesty — noncircular 92-column matrix.

We dropped melting scalars from $X$, kept static 37 °C physics + k-mers + SD–AUG,
and adopted StratifiedGroupKFold on `rfam_acc` so family leakage could not inflate AUC.

Run:
    uv run marimo edit notebooks/03_rf_honesty_noncircular_matrix.py
"""

import marimo

__generated_with = "0.24.0"
app = marimo.App(
    width="full",
    app_title="03 · RF honesty — noncircular matrix",
)


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # 03 · We made the RF honest (noncircular $X$)

    **What we did:** withheld $T_m$, Hill, $Z$, and $\Delta P_{\mathrm{RBS}}$ from the
    feature matrix, and scored with StratifiedGroupKFold grouped on Rfam accession.

    **Why:** circular melting features and family-leaky CV overstate transfer.
    This app **does not train** — it loads diagnostic JSON sidecars.

    Next: App 04 shows the honest score (GroupKFold AUC ≈ 0.28) that forced the
    pivot to EVA + yield gates.
    """
    )
    return


@app.cell
def _():
    import json
    from pathlib import Path

    import pandas as pd

    PROJECT_ROOT = Path.cwd().resolve()
    if not (PROJECT_ROOT / "src").exists():
        PROJECT_ROOT = PROJECT_ROOT.parent

    DATA = PROJECT_ROOT / "data" / "processed"
    FIG = PROJECT_ROOT / "notebooks" / "figures" / "03_rf_honesty"

    def _load(name: str):
        path = DATA / name
        if not path.exists():
            return None
        return json.loads(path.read_text())

    diag = _load("rf_noncircular_diagnostics.json")
    post = _load("rf_posthoc_report.json")
    feat = _load("rf_noncircular_feature_log.json")
    dyn = _load("refseq_dynamic_rf_diagnostics.json")
    matched = _load("refseq_matched_rf_diagnostics.json")
    ablation = _load("rf_compositional_ablation.json")

    return DATA, FIG, ablation, diag, dyn, feat, matched, pd, post


@app.cell
def _(diag, dyn, feat, mo, matched, pd):
    rows = []
    if diag is not None:
        g = diag.get("noncircular_stratified_group", {})
        s = diag.get("noncircular_stratified", {})
        rows.append(
            {
                "model": "92-col noncircular",
                "protocol": "StratifiedGroupKFold",
                "roc_auc": g.get("roc_auc"),
                "accuracy": g.get("accuracy"),
            }
        )
        rows.append(
            {
                "model": "92-col noncircular",
                "protocol": "StratifiedKFold (no groups)",
                "roc_auc": s.get("roc_auc"),
                "accuracy": s.get("accuracy"),
            }
        )
    if dyn is not None:
        rows.append(
            {
                "model": "earlier RefSeq-dynamic (sidecar)",
                "protocol": dyn.get("protocol", "see JSON"),
                "roc_auc": dyn.get("roc_auc"),
                "accuracy": dyn.get("accuracy"),
            }
        )
    if matched is not None:
        rows.append(
            {
                "model": "earlier matched (sidecar)",
                "protocol": matched.get("protocol", "see JSON"),
                "roc_auc": matched.get("roc_auc"),
                "accuracy": matched.get("accuracy"),
            }
        )

    table = pd.DataFrame(rows) if rows else pd.DataFrame()
    n_feat = None
    if feat is not None:
        n_feat = (
            feat.get("n_features")
            or feat.get("n_columns")
            or len(feat.get("feature_names", []) or [])
        )

    note = (
        f"Feature log reports **{n_feat}** columns in $X$."
        if n_feat
        else "_Feature log missing — expected `rf_noncircular_feature_log.json`._"
    )
    mo.vstack(
        [
            mo.md("## Protocol comparison (from sidecars)"),
            mo.ui.table(table) if len(table) else mo.md("_No diagnostics JSON found._"),
            mo.md(note),
        ]
    )
    return


@app.cell
def _(ablation, diag, mo, pd, post):
    perm = pd.DataFrame()
    if diag is not None:
        block = diag.get("grouped_permutation") or diag.get("permutation") or {}
        if isinstance(block, dict) and block:
            perm = pd.DataFrame(
                [
                    {"check": k, "value": v}
                    for k, v in block.items()
                    if not isinstance(v, (dict, list))
                ]
            )

    gate_note = ""
    if post is not None:
        gates = post.get("gates") or post.get("yield_gates") or {}
        if gates:
            gate_note = (
                "Post-hoc report present — melting scalars stayed out of $X$ "
                "and entered ranking / gates only."
            )

    abl_note = ""
    if ablation is not None:
        abl_note = (
            "Compositional ablation sidecar present — see App 02 for the "
            "length/%GC shortcut kill."
        )

    mo.vstack(
        [
            mo.md("## Grouped permutation (from sidecar)"),
            mo.ui.table(perm)
            if len(perm)
            else mo.md("_No permutation block in diagnostics._"),
            mo.md(gate_note) if gate_note else mo.md(""),
            mo.md(abl_note) if abl_note else mo.md(""),
            mo.md(
                """
    **Finding locked here:** honest protocol + noncircular $X$ is the thesis RF
    baseline. App 04 loads the same sidecars to show AUC ≈ 0.28 and the pivot.
    """
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
