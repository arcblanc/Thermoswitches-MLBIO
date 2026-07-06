import sys
from pathlib import Path

import pandas as pd

SRC_ROOT = Path(__file__).resolve().parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data_engineering.cd_hit_sequence_similarity import JOIN_COLUMNS
from data_engineering.paths import resolve_path
from thermo_sim.thermo_common import METADATA_COLUMNS, append_feature_table, write_feature_table

FUSED_FEATURE_COLUMNS = [
    "panel_role",
    "record_id",
    "seq_length",
    "viennarna_Tm",
    "viennarna_hill_coeff",
    "viennarna_amplitude",
    "viennarna_mean_unpaired_prob",
    "viennarna_sd_pair_prob_10C",
    "viennarna_sd_pair_prob_80C",
    "viennarna_MFE",
    "viennarna_max_loop_length",
    "viennarna_gc_content",
    "viennarna_dangles_model",
    "viennarna_fit_status",
    "nupack_Tm",
    "nupack_hill_coeff",
    "nupack_amplitude",
    "nupack_mean_exposure",
    "nupack_MFE",
    "nupack_max_stem_length",
    "nupack_max_loop_length",
    "nupack_gc_content",
    "nupack_sd_pair_prob_10C",
    "nupack_sd_pair_prob_80C",
    "nupack_fit_status",
]


def fuse_engine_features(vienna_df, nupack_df, join_on=None):
    join_on = join_on or JOIN_COLUMNS
    vienna_df = vienna_df.copy()
    nupack_df = nupack_df.copy()

    vienna_cols = [col for col in vienna_df.columns if col.startswith("viennarna_")]
    nupack_cols = [col for col in nupack_df.columns if col.startswith("nupack_")]
    meta_cols = [col for col in METADATA_COLUMNS if col in vienna_df.columns]
    extra_cols = [
        col
        for col in ("panel_role", "record_id", "seq_length")
        if col in vienna_df.columns
    ]
    left_cols = list(dict.fromkeys(join_on + meta_cols + extra_cols + vienna_cols))
    right_cols = list(dict.fromkeys(join_on + nupack_cols))

    fused = pd.merge(
        vienna_df[left_cols],
        nupack_df[right_cols],
        on=join_on,
        how="inner",
        validate="one_to_one",
    )
    if len(fused) != len(vienna_df):
        raise ValueError(
            f"Feature fusion expected {len(vienna_df)} rows but produced {len(fused)}."
        )
    return fused


def validate_fused_row(row):
    warnings = []
    role = row.get("panel_role", "")
    if role == "canonical_positive":
        for col in ("viennarna_Tm", "nupack_Tm"):
            if pd.isna(row.get(col)):
                warnings.append(f"{col} missing for canonical positive")
    if role.startswith("anomaly"):
        if row.get("viennarna_fit_status") not in {"ok", "flat", "optimize_failed"}:
            warnings.append("unexpected Vienna fit status for anomaly sequence")
    return warnings


def _fused_extra_columns(fused_df):
    return [
        col
        for col in ("panel_role", "record_id", "seq_length")
        if col in fused_df.columns
    ]


def write_fused_features(fused_df, output_csv, join_columns=None, include_label=True):
    engine_cols = [col for col in fused_df.columns if col.startswith(("viennarna_", "nupack_"))]
    return write_feature_table(
        fused_df,
        output_csv,
        engine_cols,
        extra_columns=_fused_extra_columns(fused_df),
        join_columns=join_columns,
        include_label=include_label,
    )


def append_fused_features(fused_df, output_csv, join_columns=None, include_label=True):
    engine_cols = [col for col in fused_df.columns if col.startswith(("viennarna_", "nupack_"))]
    return append_feature_table(
        fused_df,
        output_csv,
        engine_cols,
        extra_columns=_fused_extra_columns(fused_df),
        join_columns=join_columns,
        include_label=include_label,
    )
