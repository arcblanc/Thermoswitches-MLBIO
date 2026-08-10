"""Diagnostics for length-bias remediation.

Compares length-alone AUC, intensive StratifiedGroupKFold (primary),
and StratifiedKFold contrast on the length/GC-matched fused corpus.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_predict

SRC_ROOT = Path(__file__).resolve().parent.parent / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from thermo_sim.thermo_classifier import (
    LEGACY_PHYSICS_FEATURE_COLUMNS,
    PHYSICS_FEATURE_COLUMNS,
    add_intensive_features,
)

LENGTH_ALONE_AUC_GATE = 0.65


def _eval_rf(X, y, cv, groups=None):
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    if groups is None:
        proba = cross_val_predict(rf, X, y, cv=cv, method="predict_proba")[:, 1]
    else:
        # cross_val_predict supports groups=
        proba = cross_val_predict(rf, X, y, cv=cv, groups=groups, method="predict_proba")[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y, proba)),
        "accuracy": float(accuracy_score(y, pred)),
    }


def run_diagnostics(
    fused_csv="data/processed/fused_features_length_matched.csv",
    legacy_fused_csv="data/processed/fused_features.csv",
    output_json="data/processed/length_matched_rf_diagnostics.json",
    length_alone_gate=LENGTH_ALONE_AUC_GATE,
    group_col="rfam_acc",
):
    df = pd.read_csv(fused_csv)
    if group_col not in df.columns:
        raise SystemExit(f"{fused_csv} missing {group_col} — required for StratifiedGroupKFold")
    df = add_intensive_features(df)
    feature_cols = [c for c in PHYSICS_FEATURE_COLUMNS if c in df.columns]
    need = ["label", "seq_length", group_col] + feature_cols
    clean = df.dropna(subset=need).copy()
    y = clean["label"].astype(int)
    groups = clean[group_col].astype(str)

    report = {
        "n": int(len(clean)),
        "n_pos": int((y == 1).sum()),
        "n_neg": int((y == 0).sum()),
        "group_col": group_col,
        "n_families": int(groups.nunique()),
        "mean_len_pos": float(clean.loc[y == 1, "seq_length"].mean()),
        "mean_len_neg": float(clean.loc[y == 0, "seq_length"].mean()),
        "delta_mu_length": float(
            clean.loc[y == 1, "seq_length"].mean() - clean.loc[y == 0, "seq_length"].mean()
        ),
        "corr_length_viennarna_MFE": float(clean["seq_length"].corr(clean["viennarna_MFE"]))
        if "viennarna_MFE" in clean.columns
        else None,
        "corr_length_viennarna_MFE_per_nt": float(
            clean["seq_length"].corr(clean["viennarna_MFE_per_nt"])
        ),
        "corr_length_nupack_MFE": float(clean["seq_length"].corr(clean["nupack_MFE"]))
        if "nupack_MFE" in clean.columns
        else None,
        "corr_length_nupack_MFE_per_nt": float(clean["seq_length"].corr(clean["nupack_MFE_per_nt"])),
        "features_intensive": feature_cols,
    }

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    report["length_alone_stratified"] = _eval_rf(clean[["seq_length"]], y, skf)
    report["intensive_stratified_group"] = _eval_rf(
        clean[feature_cols], y, sgkf, groups=groups
    )
    report["intensive_stratified"] = _eval_rf(clean[feature_cols], y, skf)

    # Legacy reference on old fused (raw MFE, StratifiedKFold) if available
    legacy_path = Path(legacy_fused_csv)
    if legacy_path.exists():
        legacy = pd.read_csv(legacy_path)
        leg_cols = [c for c in LEGACY_PHYSICS_FEATURE_COLUMNS if c in legacy.columns]
        leg = legacy.dropna(subset=["label"] + leg_cols)
        report["legacy_raw_mfe_stratified"] = _eval_rf(
            leg[leg_cols], leg["label"].astype(int), skf
        )
        if "seq_length" in leg.columns:
            report["legacy_length_alone_stratified"] = _eval_rf(
                leg[["seq_length"]], leg["label"].astype(int), skf
            )

    length_auc = report["length_alone_stratified"]["roc_auc"]
    report["length_alone_gate"] = length_alone_gate
    report["length_alone_gate_passed"] = bool(length_auc <= length_alone_gate)
    if not report["length_alone_gate_passed"]:
        report["warning"] = (
            f"Length-alone AUC {length_auc:.3f} exceeds gate {length_alone_gate}; "
            "matching may be insufficient — do not trust RF as biophysical."
        )

    out = Path(output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nSaved → {out}")
    if not report["length_alone_gate_passed"]:
        raise SystemExit(2)
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fused-csv", default="data/processed/fused_features_length_matched.csv")
    p.add_argument("--output-json", default="data/processed/length_matched_rf_diagnostics.json")
    p.add_argument("--length-alone-gate", type=float, default=LENGTH_ALONE_AUC_GATE)
    p.add_argument(
        "--group-col",
        default="rfam_acc",
        help="Column for StratifiedGroupKFold (rfam_acc; RefSeq negs use REFSEQ:assembly).",
    )
    args = p.parse_args()
    run_diagnostics(
        fused_csv=args.fused_csv,
        output_json=args.output_json,
        length_alone_gate=args.length_alone_gate,
        group_col=args.group_col,
    )


if __name__ == "__main__":
    main()
