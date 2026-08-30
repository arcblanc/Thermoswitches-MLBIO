"""Post-hoc gates on RF scores: confidence bins, melting filters, consensus tests.

These scalars are not Random Forest inputs. EVA 2000 yield remains
Z ≤ -2 ∧ ΔP_RBS > 0 ∧ E_Rfam > 1e-3 until generated FASTA has Hill/Tm.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import cast

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, mannwhitneyu, spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    StratifiedGroupKFold,
    StratifiedKFold,
    cross_val_predict,
)

from data_engineering.paths import resolve_path

HIGH_CONF = 0.80
MID_LO = 0.40
MID_HI = 0.60
LOW_CONF = 0.20
HILL_GATE = 1.0
HILL_SNAP = 1.5
TM_MIN = 42.0
TM_MAX = 45.0
AMPLITUDE_MIN = 0.50
Z_MAX = -2.0
DELTA_P_MIN = 0.0
SPEARMAN_MIN_N = 25
# Baseline RBS repression: unpaired probability near zero at 37 °C.
P_OPEN_LOCKED_MAX = 0.20

DEFAULT_POSTHOC_JSON = "data/processed/rf_posthoc_report.json"
DEFAULT_GROUP_COL = "rfam_acc"

SPEARMAN_PAIRS = (
    ("viennarna_Tm", "nupack_Tm"),
    ("viennarna_hill_coeff", "nupack_hill_coeff"),
    ("viennarna_amplitude", "nupack_amplitude"),
)

BIN_TEST_COLUMNS = [
    "viennarna_Tm",
    "viennarna_hill_coeff",
    "viennarna_amplitude",
    "viennarna_mfe_zscore",
    "viennarna_delta_P_RBS",
]


def confidence_bin(yhat: float) -> str:
    """Map an OOF probability to a high/mid/low confidence bin."""
    if yhat >= HIGH_CONF:
        return "high"
    if yhat <= LOW_CONF:
        return "low"
    if MID_LO < yhat < MID_HI:
        return "mid"
    return "other"


def confidence_bins(yhat: np.ndarray | pd.Series) -> pd.Series:
    """Map each OOF probability to a high/mid/low/other confidence bin."""
    return pd.Series(yhat).map(confidence_bin)


def _gate_float(value: object) -> float | None:
    """Return a finite float gate operand, or None when missing."""
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (int, float, str)):
        return float(value)
    return None


def gate_delta_p_rbs(delta_p: object) -> bool:
    """Return True when ΔP_RBS is defined and strictly positive."""
    operand = _gate_float(delta_p)
    return operand is not None and operand > DELTA_P_MIN


def gate_hill(n_h: object, *, threshold: float = HILL_GATE) -> bool:
    """Return True when the Hill coefficient exceeds the threshold."""
    operand = _gate_float(n_h)
    return operand is not None and operand > threshold


def gate_tm(tm: object, *, lo: float = TM_MIN, hi: float = TM_MAX) -> bool:
    """Return True when Tm falls inside the inclusive temperature window."""
    operand = _gate_float(tm)
    return operand is not None and lo <= operand <= hi


def gate_zscore(z: object, *, z_max: float = Z_MAX) -> bool:
    """Return True when the MFE Z-score is at or below z_max."""
    operand = _gate_float(z)
    return operand is not None and operand <= z_max


def gate_amplitude(amp: object, *, minimum: float = AMPLITUDE_MIN) -> bool:
    """Return True when melting amplitude meets the minimum dynamic range."""
    operand = _gate_float(amp)
    return operand is not None and operand >= minimum


def gate_baseline_repression(
    p_open: object, *, maximum: float = P_OPEN_LOCKED_MAX
) -> bool:
    """Return True when 37 °C RBS unpaired probability is locked down."""
    operand = _gate_float(p_open)
    return operand is not None and operand <= maximum


def visual_checklist_flags(row: pd.Series) -> dict[str, bool]:
    """Score a row against the visual melting-curve checklist gates."""
    n_h = row.get("viennarna_hill_coeff")
    tm = row.get("viennarna_Tm")
    amp = row.get("viennarna_amplitude")
    p_open = row.get("viennarna_P_open_RBS_37")
    if p_open is None or (isinstance(p_open, float) and np.isnan(p_open)):
        p_open = row.get("viennarna_hill_bottom")
    n_h_val = _gate_float(n_h)
    return {
        "sigmoidal_steepness_snap": gate_hill(n_h, threshold=HILL_SNAP),
        "sigmoidal_ramp": n_h_val is not None and n_h_val <= 1.0,
        "inflection_tm": gate_tm(tm),
        "dynamic_range": gate_amplitude(amp),
        "baseline_repression": gate_baseline_repression(p_open),
    }


def _spearman_pair(a: pd.Series, b: pd.Series) -> dict[str, object]:
    """Compute Spearman correlation for one Vienna vs NUPACK feature pair."""
    mask = a.notna() & b.notna()
    n = int(mask.sum())
    if n < 3:
        return {"n": n, "r_s": None, "p_value": None, "underpowered": True}
    r, p = spearmanr(a[mask], b[mask])
    return {
        "n": n,
        "r_s": None if pd.isna(r) else float(r),
        "p_value": None if pd.isna(p) else float(p),
        "underpowered": False,
    }


def spearman_consensus(
    df: pd.DataFrame,
    *,
    mask: pd.Series | None = None,
    min_n: int = SPEARMAN_MIN_N,
    require_min_n: bool = False,
) -> dict:
    """Vienna vs NUPACK Spearman. High-bin calls should set require_min_n=True."""
    subset = df if mask is None else df.loc[mask]
    pairs = {}
    ns = []
    for v_col, n_col in SPEARMAN_PAIRS:
        if v_col not in subset.columns or n_col not in subset.columns:
            pairs[f"{v_col}_vs_{n_col}"] = {
                "n": 0,
                "r_s": None,
                "p_value": None,
                "underpowered": True,
                "missing_columns": True,
            }
            ns.append(0)
            continue
        stats = _spearman_pair(subset[v_col], subset[n_col])
        ns.append(int(cast(int, stats["n"])))
        pairs[f"{v_col}_vs_{n_col}"] = stats
    n_complete = int(min(ns) if ns else 0)
    underpowered = bool(require_min_n and n_complete < min_n)
    if underpowered:
        for key in pairs:
            pairs[key] = {
                "n": pairs[key]["n"],
                "r_s": None,
                "p_value": None,
                "underpowered": True,
                "min_n": min_n,
            }
    return {
        "n_complete": n_complete,
        "min_n": min_n,
        "underpowered": underpowered,
        "pairs": pairs,
    }


def two_sample_tests(high: pd.Series, low: pd.Series) -> dict:
    """Run Mann–Whitney U and KS tests comparing high vs low confidence bins."""
    high = pd.to_numeric(high, errors="coerce").dropna()
    low = pd.to_numeric(low, errors="coerce").dropna()
    if len(high) < 2 or len(low) < 2:
        return {
            "n_high": int(len(high)),
            "n_low": int(len(low)),
            "mannwhitney_u": None,
            "mannwhitney_p": None,
            "ks_statistic": None,
            "ks_p": None,
        }
    u_stat, u_p = mannwhitneyu(high, low, alternative="two-sided")
    ks_stat, ks_p = ks_2samp(high, low)
    return {
        "n_high": int(len(high)),
        "n_low": int(len(low)),
        "mannwhitney_u": float(u_stat),
        "mannwhitney_p": float(u_p),
        "ks_statistic": float(ks_stat),
        "ks_p": float(ks_p),
    }


def _oof_proba(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series | None,
    random_state: int = 42,
    n_estimators: int = 200,
) -> np.ndarray:
    """Compute out-of-fold positive-class probabilities for a Random Forest."""
    rf = RandomForestClassifier(
        n_estimators=n_estimators, random_state=random_state, n_jobs=-1
    )
    if groups is not None:
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_state)
        return cross_val_predict(
            rf, X, y, cv=cv, groups=groups, method="predict_proba"
        )[:, 1]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    return cross_val_predict(rf, X, y, cv=cv, method="predict_proba")[:, 1]


def evaluate_posthoc(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    group_col: str = DEFAULT_GROUP_COL,
    yhat: np.ndarray | None = None,
    output_json: str = DEFAULT_POSTHOC_JSON,
    n_estimators: int = 200,
    random_state: int = 42,
) -> dict:
    """Write post-hoc confidence bins, melting gates, and consensus tests."""
    need = ["label"] + [c for c in feature_cols if c in df.columns]
    work = df.dropna(subset=need).copy()
    y = work["label"].astype(int)
    X = work[feature_cols]
    groups = work[group_col].astype(str) if group_col in work.columns else None

    if yhat is None:
        yhat = _oof_proba(
            X, y, groups, random_state=random_state, n_estimators=n_estimators
        )
    work = work.copy()
    work["yhat"] = np.asarray(yhat, dtype=float)
    work["confidence_bin"] = confidence_bins(work["yhat"])

    high_mask = work["confidence_bin"] == "high"
    mid_mask = work["confidence_bin"] == "mid"
    low_mask = work["confidence_bin"] == "low"
    n_high = int(high_mask.sum())

    if "viennarna_delta_P_RBS" in work.columns:
        dp = work["viennarna_delta_P_RBS"]
    else:
        if {"viennarna_P_open_RBS_55", "viennarna_P_open_RBS_37"} <= set(work.columns):
            dp = work["viennarna_P_open_RBS_55"] - work["viennarna_P_open_RBS_37"]
        else:
            dp = pd.Series(np.nan, index=work.index)

    gates = {
        "delta_P_RBS_gt_0": int(dp.map(gate_delta_p_rbs).sum()),
        "hill_gt_1": int(
            work.get("viennarna_hill_coeff", pd.Series(dtype=float))
            .map(gate_hill)
            .sum()
        )
        if "viennarna_hill_coeff" in work.columns
        else None,
        "nupack_hill_gt_1": int(work["nupack_hill_coeff"].map(gate_hill).sum())
        if "nupack_hill_coeff" in work.columns
        else None,
        "tm_42_45": int(work["viennarna_Tm"].map(gate_tm).sum())
        if "viennarna_Tm" in work.columns
        else None,
        "z_le_minus2": int(work["viennarna_mfe_zscore"].map(gate_zscore).sum())
        if "viennarna_mfe_zscore" in work.columns
        else None,
    }

    checklist_counts = {
        "sigmoidal_steepness_snap": 0,
        "sigmoidal_ramp": 0,
        "inflection_tm": 0,
        "dynamic_range": 0,
        "baseline_repression": 0,
        "all_four_pass": 0,
    }
    for _, row in work.iterrows():
        flags = visual_checklist_flags(row)
        for k, v in flags.items():
            if k in checklist_counts and v:
                checklist_counts[k] += 1
        if (
            flags["sigmoidal_steepness_snap"]
            and flags["inflection_tm"]
            and flags["dynamic_range"]
            and flags["baseline_repression"]
        ):
            checklist_counts["all_four_pass"] += 1

    bin_tests = {}
    for col in BIN_TEST_COLUMNS:
        if col not in work.columns:
            continue
        bin_tests[col] = two_sample_tests(
            work.loc[high_mask, col], work.loc[low_mask, col]
        )

    has_hill = (
        "viennarna_hill_coeff" in work.columns
        and work["viennarna_hill_coeff"].notna().any()
    )
    spearman_panel = spearman_consensus(work, require_min_n=False)
    spearman_high = spearman_consensus(
        work, mask=high_mask, min_n=SPEARMAN_MIN_N, require_min_n=True
    )
    report = {
        "written_at": datetime.now(timezone.utc).isoformat(),
        "n": int(len(work)),
        "n_pos": int((y == 1).sum()),
        "n_neg": int((y == 0).sum()),
        "group_col": group_col if groups is not None else None,
        "bins": {
            "high_ge_0.80": n_high,
            "mid_0.40_0.60": int(mid_mask.sum()),
            "low_le_0.20": int(low_mask.sum()),
            "other": int((work["confidence_bin"] == "other").sum()),
        },
        "n_high": n_high,
        "gates": gates,
        "spearman_panel_wide_primary": spearman_panel,
        "spearman_high_bin": spearman_high,
        "high_bin_spearman_underpowered": bool(spearman_high["underpowered"]),
        "bin_tests_high_vs_low": bin_tests,
        "visual_checklist": checklist_counts,
        "visual_checklist_thresholds": {
            "n_H_snap": HILL_SNAP,
            "n_H_ramp": 1.0,
            "n_H_posthoc_gate": HILL_GATE,
            "Tm_C": [TM_MIN, TM_MAX],
            "delta_theta_min": AMPLITUDE_MIN,
            "P_open_RBS_37_max": P_OPEN_LOCKED_MAX,
        },
        "eva_hill_not_available": (
            "EVA stream FASTA triage still uses Z / ΔP_RBS / E_Rfam only. "
            "Full post-hoc Hill/Tm gates need thermo_batch on generated sequences."
        ),
        "has_hill_columns": bool(has_hill),
    }
    path = resolve_path(output_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"Wrote {path}")
    return report
