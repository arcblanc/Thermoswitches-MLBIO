"""Monotonic XGBoost diagnostics on RefSeq-dynamic fused features.

Compares length-alone StratifiedKFold, intensive StratifiedGroupKFold (primary),
and StratifiedKFold, side-by-side with unconstrained RF GroupKFold baseline.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import (
    StratifiedGroupKFold,
    StratifiedKFold,
    cross_val_predict,
)

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from thermo_sim.thermo_classifier import (
    MONOTONE_CONSTRAINTS_BY_FEATURE,
    PHYSICS_FEATURE_COLUMNS,
    add_intensive_features,
    build_xgboost_monotonic,
    monotone_constraints_tuple,
)

LENGTH_ALONE_AUC_GATE = 0.65


def _eval_proba(
    estimator: object,
    X: pd.DataFrame,
    y: pd.Series,
    cv: StratifiedGroupKFold | StratifiedKFold,
    groups: pd.Series | None = None,
) -> dict[str, float]:
    """Cross-validate predict_proba and return ROC-AUC and accuracy."""
    if groups is None:
        proba = cross_val_predict(estimator, X, y, cv=cv, method="predict_proba")[:, 1]
    else:
        proba = cross_val_predict(
            estimator, X, y, cv=cv, groups=groups, method="predict_proba"
        )[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y, proba)),
        "accuracy": float(accuracy_score(y, pred)),
    }


def _eval_rf(
    X: pd.DataFrame,
    y: pd.Series,
    cv: StratifiedGroupKFold | StratifiedKFold,
    groups: pd.Series | None = None,
) -> dict[str, float]:
    """Cross-validate an unconstrained RF and return ROC-AUC and accuracy."""
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    return _eval_proba(rf, X, y, cv, groups=groups)


def _eval_xgb(
    X: pd.DataFrame,
    y: pd.Series,
    feature_cols: list[str],
    cv: StratifiedGroupKFold | StratifiedKFold,
    groups: pd.Series | None = None,
) -> dict[str, float]:
    """Cross-validate monotonic XGBoost and return ROC-AUC and accuracy."""
    model, _ = build_xgboost_monotonic(feature_cols)
    return _eval_proba(model, X, y, cv, groups=groups)


def run_diagnostics(
    fused_csv: str = "data/processed/fused_features_refseq_dynamic.csv",
    output_json: str = "data/processed/xgb_refseq_dynamic_diagnostics.json",
    rf_baseline_json: str = "data/processed/refseq_dynamic_rf_diagnostics.json",
    length_alone_gate: float = LENGTH_ALONE_AUC_GATE,
    group_col: str = "rfam_acc",
) -> dict[str, object]:
    """Compare length-alone RF, monotonic XGB, and unconstrained RF GroupKFold."""
    df = pd.read_csv(fused_csv)
    if group_col not in df.columns:
        raise SystemExit(
            f"{fused_csv} missing {group_col} — required for StratifiedGroupKFold"
        )
    df = add_intensive_features(df)
    feature_cols = [c for c in PHYSICS_FEATURE_COLUMNS if c in df.columns]
    need = ["label", "seq_length", group_col] + feature_cols
    clean = df.dropna(subset=need).copy()
    y = clean["label"].astype(int)
    groups = clean[group_col].astype(str)
    constraints = monotone_constraints_tuple(feature_cols)

    report = {
        "n": int(len(clean)),
        "n_pos": int((y == 1).sum()),
        "n_neg": int((y == 0).sum()),
        "group_col": group_col,
        "n_families": int(groups.nunique()),
        "features_intensive": feature_cols,
        "monotone_constraints": {
            c: int(MONOTONE_CONSTRAINTS_BY_FEATURE.get(c, 0)) for c in feature_cols
        },
        "monotone_constraints_tuple": list(constraints),
    }

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    report["length_alone_stratified"] = _eval_rf(clean[["seq_length"]], y, skf)
    report["xgb_monotonic_stratified_group"] = _eval_xgb(
        clean[feature_cols], y, feature_cols, sgkf, groups=groups
    )
    report["xgb_monotonic_stratified"] = _eval_xgb(
        clean[feature_cols], y, feature_cols, skf
    )
    report["rf_unconstrained_stratified_group"] = _eval_rf(
        clean[feature_cols], y, sgkf, groups=groups
    )

    rf_group_auc = report["rf_unconstrained_stratified_group"]["roc_auc"]
    baseline_path = Path(rf_baseline_json)
    if baseline_path.exists():
        baseline = json.loads(baseline_path.read_text())
        report["rf_baseline_json"] = str(baseline_path)
        report["rf_baseline_groupkfold_auc"] = float(
            baseline.get("intensive_stratified_group", {}).get("roc_auc", rf_group_auc)
        )
        # Prefer measured side-by-side RF on the same rows for delta; keep baseline for reference.
        rf_group_auc = float(report["rf_unconstrained_stratified_group"]["roc_auc"])

    xgb_group_auc = report["xgb_monotonic_stratified_group"]["roc_auc"]
    length_auc = report["length_alone_stratified"]["roc_auc"]
    report["length_alone_gate"] = length_alone_gate
    report["length_alone_gate_passed"] = bool(length_auc <= length_alone_gate)
    report["decision_matrix"] = {
        "rf_groupkfold_auc": round(rf_group_auc, 4),
        "xgb_monotonic_groupkfold_auc": round(xgb_group_auc, 4),
        "delta_auc_vs_rf": round(xgb_group_auc - rf_group_auc, 4),
        "length_alone_gate_passed": report["length_alone_gate_passed"],
    }

    out = Path(output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["decision_matrix"], indent=2))
    print(f"Wrote {out}")
    return report


def main() -> None:
    """Parse CLI args and run monotonic XGBoost diagnostics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fused-csv",
        default="data/processed/fused_features_refseq_dynamic.csv",
    )
    parser.add_argument(
        "--output-json",
        default="data/processed/xgb_refseq_dynamic_diagnostics.json",
    )
    parser.add_argument(
        "--rf-baseline-json",
        default="data/processed/refseq_dynamic_rf_diagnostics.json",
    )
    parser.add_argument("--group-col", default="rfam_acc")
    parser.add_argument(
        "--length-alone-gate", type=float, default=LENGTH_ALONE_AUC_GATE
    )
    args = parser.parse_args()
    run_diagnostics(
        fused_csv=args.fused_csv,
        output_json=args.output_json,
        rf_baseline_json=args.rf_baseline_json,
        length_alone_gate=args.length_alone_gate,
        group_col=args.group_col,
    )


if __name__ == "__main__":
    main()
