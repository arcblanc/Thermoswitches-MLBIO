"""Non-circular Random Forest inputs: static 37 °C physics, k-mers, SD–AUG.

Melting phenotype (Tm, Hill, amplitude, Z, ΔP_RBS, ΔΔG) is excluded from X
and scored only as post-hoc gates.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from data_engineering.cd_hit_sequence_similarity import JOIN_COLUMNS
from data_engineering.paths import resolve_path
from thermo_sim.thermo_common import (
    detect_shine_dalgarno,
    load_balanced_dataset,
    load_fasta_dataset,
    normalize_sequence,
)

RNA_ALPHABET = ("A", "U", "G", "C")
DINUCLEOTIDES = [a + b for a in RNA_ALPHABET for b in RNA_ALPHABET]
TRINUCLEOTIDES = [
    a + b + c for a in RNA_ALPHABET for b in RNA_ALPHABET for c in RNA_ALPHABET
]
DINUC_COLUMNS = [f"dinuc_{k}" for k in DINUCLEOTIDES]
TRINUC_COLUMNS = [f"trinuc_{k}" for k in TRINUCLEOTIDES]

SD_AUG_SENTINEL = -1
SD_AUG_COLUMNS = ["sd_aug_spacing", "sd_aug_missing"]

STATIC_BIOPHYSICS_COLUMNS = [
    "viennarna_MFE_per_nt",
    "nupack_MFE_per_nt",
    "viennarna_ensemble_diversity",
    "viennarna_mean_positional_entropy",
    "nupack_max_stem_length",
    "nupack_max_loop_length",
    "viennarna_max_loop_length",
    "viennarna_max_stem_length",
]

COMPOSITION_COLUMNS = [
    "seq_length",
    "viennarna_gc_content",
    "nupack_gc_content",
    "P_paired_RBS_37",
]

CIRCULAR_EXCLUDED_COLUMNS = [
    "nupack_Tm",
    "nupack_hill_coeff",
    "nupack_amplitude",
    "viennarna_Tm",
    "viennarna_hill_coeff",
    "viennarna_amplitude",
    "viennarna_mfe_zscore",
    "viennarna_delta_P_RBS",
    "viennarna_delta_delta_G",
]

DEFAULT_DATASET_CSV = "data/processed/balanced/length_gc_matched_refseq_dataset.csv"
DEFAULT_DATASET_FASTA = "data/processed/balanced/length_gc_matched_refseq_dataset.fasta"
DEFAULT_FEATURE_LOG = "data/processed/rf_noncircular_feature_log.json"

# Physics NaNs may drop a row; SD–AUG sentinels never do.
PHYSICS_DROPNA_CANDIDATES = [
    "viennarna_MFE_per_nt",
    "nupack_MFE_per_nt",
    "viennarna_ensemble_diversity",
    "viennarna_mean_positional_entropy",
    "nupack_max_stem_length",
    "nupack_max_loop_length",
    "viennarna_max_loop_length",
    "seq_length",
    "P_paired_RBS_37",
]


def kmer_frequency_columns(k: int) -> list[str]:
    """Return dinucleotide or trinucleotide feature column names."""
    if k == 2:
        return list(DINUC_COLUMNS)
    if k == 3:
        return list(TRINUC_COLUMNS)
    raise ValueError(f"Unsupported k={k}; expected 2 or 3")


def kmer_frequencies(sequence: str, k: int) -> dict[str, float]:
    """Intensive k-mer frequencies (counts / (n-k+1)); missing windows → 0."""
    seq = normalize_sequence(sequence)
    keys = DINUCLEOTIDES if k == 2 else TRINUCLEOTIDES
    prefix = "dinuc_" if k == 2 else "trinuc_"
    n = len(seq)
    denom = n - k + 1
    counts = {prefix + mer: 0.0 for mer in keys}
    if denom <= 0:
        return counts
    for i in range(denom):
        mer = seq[i : i + k]
        col = prefix + mer
        if col in counts:
            counts[col] += 1.0
    return {col: val / denom for col, val in counts.items()}


def sd_aug_features(sequence: str) -> dict[str, int]:
    """SD-to-AUG spacer. Missing AUG → sentinel -1 plus sd_aug_missing=1 (never NaN)."""
    seq = normalize_sequence(sequence)
    if not seq:
        return {"sd_aug_spacing": SD_AUG_SENTINEL, "sd_aug_missing": 1}
    _sd_start, sd_end = detect_shine_dalgarno(seq)
    aug = seq.find("AUG", sd_end + 1)
    if aug < 0:
        aug = seq.rfind("AUG")
        if aug < 0 or aug <= sd_end:
            return {"sd_aug_spacing": SD_AUG_SENTINEL, "sd_aug_missing": 1}
    return {"sd_aug_spacing": int(aug - sd_end - 1), "sd_aug_missing": 0}


def p_paired_rbs_37(p_open: object) -> float | None:
    """Convert RBS unpaired probability at 37 °C to paired probability."""
    if p_open is None or (isinstance(p_open, float) and np.isnan(p_open)):
        return None
    if isinstance(p_open, (int, float, str)):
        return float(1.0 - float(p_open))
    return None


def attach_sequences(
    df: pd.DataFrame,
    *,
    dataset_csv: str = DEFAULT_DATASET_CSV,
    dataset_fasta: str = DEFAULT_DATASET_FASTA,
    denovo_fasta: str | None = None,
) -> pd.DataFrame:
    """Join RNA sequences onto a fused table without requiring a sequence column."""
    out = df.copy()
    if "sequence" in out.columns and out["sequence"].notna().any():
        out["sequence"] = out["sequence"].map(
            lambda s: normalize_sequence(s) if pd.notna(s) else s
        )
        if out["sequence"].notna().all():
            return out

    joined = False
    if all(c in out.columns for c in JOIN_COLUMNS):
        csv_path = resolve_path(dataset_csv)
        fa_path = resolve_path(dataset_fasta)
        if csv_path.exists() and fa_path.exists():
            panel = load_balanced_dataset(dataset_csv, dataset_fasta)
            for c in ("seq_start", "seq_end"):
                out[c] = out[c].astype(int)
                panel[c] = panel[c].astype(int)
            seq_map = panel[JOIN_COLUMNS + ["sequence"]].drop_duplicates(JOIN_COLUMNS)
            out = out.merge(
                seq_map, on=JOIN_COLUMNS, how="left", suffixes=("", "_panel")
            )
            if "sequence_panel" in out.columns:
                out["sequence"] = out["sequence"].where(
                    out["sequence"].notna(), out["sequence_panel"]
                )
                out = out.drop(columns=["sequence_panel"])
            joined = out["sequence"].notna().any()

    if (not joined or out["sequence"].isna().any()) and denovo_fasta:
        fa_path = resolve_path(denovo_fasta)
        if fa_path.exists() and "record_id" in out.columns:
            fasta_df = load_fasta_dataset(denovo_fasta)
            out = out.merge(
                fasta_df[["record_id", "sequence"]].rename(
                    columns={"sequence": "_fa_seq"}
                ),
                on="record_id",
                how="left",
            )
            if "sequence" not in out.columns:
                out["sequence"] = out["_fa_seq"]
            else:
                out["sequence"] = out["sequence"].where(
                    out["sequence"].notna(), out["_fa_seq"]
                )
            out = out.drop(columns=["_fa_seq"])
            joined = out["sequence"].notna().any()

    if "sequence" not in out.columns or not out["sequence"].notna().any():
        raise ValueError(
            "Could not attach sequences. Provide length_gc_matched_refseq_dataset "
            "CSV/FASTA (labelled panel) or a de novo FASTA with record_id."
        )
    return out


def add_composition_features(df: pd.DataFrame) -> pd.DataFrame:
    """k-mer frequencies, SD–AUG sentinel pair, and P_paired_RBS_37. Never drops rows."""
    if "sequence" not in df.columns:
        raise ValueError("sequence column required for composition features")
    out = df.copy()
    dinuc_rows = []
    trinuc_rows = []
    sd_rows = []
    for seq in out["sequence"].fillna(""):
        dinuc_rows.append(kmer_frequencies(seq, 2))
        trinuc_rows.append(kmer_frequencies(seq, 3))
        sd_rows.append(sd_aug_features(seq))
    dinuc_df = pd.DataFrame(dinuc_rows, index=out.index)
    trinuc_df = pd.DataFrame(trinuc_rows, index=out.index)
    sd_df = pd.DataFrame(sd_rows, index=out.index)
    for col in dinuc_df.columns:
        out[col] = dinuc_df[col]
    for col in trinuc_df.columns:
        out[col] = trinuc_df[col]
    out["sd_aug_spacing"] = sd_df["sd_aug_spacing"].astype(int)
    out["sd_aug_missing"] = sd_df["sd_aug_missing"].astype(int)

    if "viennarna_P_open_RBS_37" in out.columns:
        out["P_paired_RBS_37"] = 1.0 - out["viennarna_P_open_RBS_37"].astype(float)
    elif "P_paired_RBS_37" not in out.columns:
        out["P_paired_RBS_37"] = np.nan
    return out


def noncircular_feature_columns(df: pd.DataFrame) -> list[str]:
    """Ordered X columns that are present on this frame."""
    cols: list[str] = []
    for col in STATIC_BIOPHYSICS_COLUMNS + COMPOSITION_COLUMNS:
        if col in df.columns and col not in cols:
            cols.append(col)
    # Prefer Vienna GC; keep NUPACK GC only if Vienna is absent.
    if "viennarna_gc_content" in cols and "nupack_gc_content" in cols:
        cols.remove("nupack_gc_content")
    cols.extend([c for c in DINUC_COLUMNS if c in df.columns])
    cols.extend([c for c in TRINUC_COLUMNS if c in df.columns])
    cols.extend([c for c in SD_AUG_COLUMNS if c in df.columns])
    return cols


def physics_dropna_columns(df: pd.DataFrame) -> list[str]:
    """Return physics columns whose NaNs should drop training rows."""
    return [c for c in PHYSICS_DROPNA_CANDIDATES if c in df.columns]


def build_noncircular_matrix(
    df: pd.DataFrame,
    *,
    dataset_csv: str = DEFAULT_DATASET_CSV,
    dataset_fasta: str = DEFAULT_DATASET_FASTA,
    denovo_fasta: str | None = None,
    already_intensive: bool = False,
) -> pd.DataFrame:
    """Attach sequences, intensive physics, k-mers, SD–AUG, P_paired."""
    from thermo_sim.thermo_classifier import add_intensive_features

    out = df.copy()
    if not already_intensive:
        if "seq_length" not in out.columns and "sequence" in out.columns:
            out["seq_length"] = out["sequence"].astype(str).str.len()
        out = add_intensive_features(out)
    out = attach_sequences(
        out,
        dataset_csv=dataset_csv,
        dataset_fasta=dataset_fasta,
        denovo_fasta=denovo_fasta,
    )
    out = add_composition_features(out)
    return out


def feature_groups(feature_cols: list[str]) -> dict[str, list[str]]:
    """Blocks for grouped permutation importance (not per-k-mer MDI)."""
    static = [c for c in STATIC_BIOPHYSICS_COLUMNS if c in feature_cols]
    composition = [c for c in COMPOSITION_COLUMNS if c in feature_cols]
    dinuc = [c for c in DINUC_COLUMNS if c in feature_cols]
    trinuc = [c for c in TRINUC_COLUMNS if c in feature_cols]
    sd_aug = [c for c in SD_AUG_COLUMNS if c in feature_cols]
    groups = {
        "static_biophysics": static,
        "composition": composition,
        "dinucleotides": dinuc,
        "trinucleotides": trinuc,
        "sd_aug": sd_aug,
    }
    return {name: cols for name, cols in groups.items() if cols}


def grouped_permutation_importance(
    model: RandomForestClassifier,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    groups: dict[str, list[str]] | None = None,
    n_repeats: int = 5,
    random_state: int = 42,
) -> dict:
    """AUC drop when shuffling an entire feature block (same permutation per group)."""
    feature_cols = list(X.columns)
    groups = groups or feature_groups(feature_cols)
    y_arr = np.asarray(y)
    class_labels = [int(c) for c in np.asarray(model.classes_).tolist()]
    pos = class_labels.index(1) if 1 in class_labels else 0
    baseline_proba = model.predict_proba(X)[:, pos]
    baseline = float(roc_auc_score(y_arr, baseline_proba))
    rng = np.random.RandomState(random_state)
    report: dict = {"baseline_auc": baseline, "n_repeats": n_repeats, "groups": {}}
    for name, cols in groups.items():
        present = [c for c in cols if c in X.columns]
        if not present:
            continue
        drops = []
        for _ in range(int(n_repeats)):
            Xp = X.copy()
            perm = rng.permutation(len(Xp))
            Xp.loc[:, present] = Xp.iloc[perm][present].to_numpy()
            proba = model.predict_proba(Xp)[:, pos]
            drops.append(baseline - float(roc_auc_score(y_arr, proba)))
        report["groups"][name] = {
            "n_features": int(len(present)),
            "columns": present,
            "mean_auc_drop": float(np.mean(drops)),
            "std_auc_drop": float(np.std(drops, ddof=0)),
        }
    return report


def aug_missing_by_class(df: pd.DataFrame) -> dict:
    """Count SD–AUG missing sentinels overall and split by class label."""
    stats = {
        "n_missing_aug": int((df["sd_aug_missing"] == 1).sum())
        if "sd_aug_missing" in df.columns
        else 0,
        "n_missing_aug_pos": None,
        "n_missing_aug_neg": None,
    }
    if "sd_aug_missing" not in df.columns or "label" not in df.columns:
        return stats
    y = df["label"].astype(int)
    stats["n_missing_aug_pos"] = int(((df["sd_aug_missing"] == 1) & (y == 1)).sum())
    stats["n_missing_aug_neg"] = int(((df["sd_aug_missing"] == 1) & (y == 0)).sum())
    stats["n_pos"] = int((y == 1).sum())
    stats["n_neg"] = int((y == 0).sum())
    return stats


def write_feature_log(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    output_json: str = DEFAULT_FEATURE_LOG,
    dataset_fasta: str = DEFAULT_DATASET_FASTA,
    extra: dict | None = None,
) -> dict:
    """Write the noncircular feature inventory JSON sidecar and return it."""
    log = {
        "written_at": datetime.now(timezone.utc).isoformat(),
        "n_rows": int(len(df)),
        "n_features": int(len(feature_cols)),
        "feature_columns": feature_cols,
        "feature_groups": feature_groups(feature_cols),
        "circular_excluded": CIRCULAR_EXCLUDED_COLUMNS,
        "dinucleotides": DINUCLEOTIDES,
        "trinucleotides": TRINUCLEOTIDES,
        "sd_aug_sentinel": SD_AUG_SENTINEL,
        "dataset_fasta": str(resolve_path(dataset_fasta)),
        "aug_missing": aug_missing_by_class(df),
        "grouped_importance_definition": (
            "Grouped permutation: shuffle all columns in a block with one shared "
            "row permutation; report ROC-AUC drop. Do not use Gini/MDI."
        ),
    }
    if extra:
        log.update(extra)
    path = resolve_path(output_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(log, indent=2))
    return log
