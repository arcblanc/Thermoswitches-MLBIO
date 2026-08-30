#!/usr/bin/env python3
"""Merge ENN-panel Rfam positives with RUS-matched negative fused rows.

Hard requirements (plan G2/G3):
- Exact column-set parity between positive and negative panels.
- Negative rfam_acc values remain REFSEQ:{assembly} with nunique > 1.

Negatives from thermo_batch may include newer Hill bottom/top / max_stem columns
that the ENN positive panel lacks; those are dropped before the equality assert.
Metadata rfam_acc / rfam_id are joined from the matched-negatives CSV.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data_engineering.paths import resolve_path

JOIN_COLUMNS = ["rfamseq_acc", "seq_start", "seq_end"]
DEFAULT_POS_FUSED = "data/processed/fused_features_refseq_dynamic.csv"
DEFAULT_NEG_FUSED = "data/processed/fused_features_refseq_rus_neg_only.csv"
DEFAULT_NEG_META = "data/processed/balanced/length_gc_matched_refseq_rus_negatives.csv"
DEFAULT_OUT = "data/processed/fused_features_refseq_dynamic_rus.csv"


def _attach_group_metadata(df_neg: pd.DataFrame, meta_csv: Path) -> pd.DataFrame:
    """Join rfam_acc / rfam_id from the matched-negatives metadata table."""
    meta = pd.read_csv(meta_csv)
    for col in ("seq_start", "seq_end"):
        meta[col] = meta[col].astype(int)
        df_neg[col] = df_neg[col].astype(int)
    keep = JOIN_COLUMNS + [c for c in ("rfam_acc", "rfam_id") if c in meta.columns]
    meta = meta[keep].drop_duplicates(JOIN_COLUMNS)
    out = df_neg.drop(
        columns=[c for c in ("rfam_acc", "rfam_id") if c in df_neg.columns]
    )
    out = out.merge(meta, on=JOIN_COLUMNS, how="left")
    if out["rfam_acc"].isna().any():
        raise AssertionError("Failed to attach rfam_acc to some negative rows")
    return out


def merge_rus_fused_panel(
    *,
    positives_fused: str = DEFAULT_POS_FUSED,
    negatives_fused: str = DEFAULT_NEG_FUSED,
    negatives_meta: str = DEFAULT_NEG_META,
    output_csv: str = DEFAULT_OUT,
) -> Path:
    """Concatenate untouched positives with new RUS negatives after schema checks."""
    pos_path = resolve_path(positives_fused)
    neg_path = resolve_path(negatives_fused)
    meta_path = resolve_path(negatives_meta)
    out_path = resolve_path(output_csv)

    df_pos = pd.read_csv(pos_path)
    df_neg = pd.read_csv(neg_path)
    df_pos = df_pos.loc[df_pos["label"].astype(int) == 1].copy()
    df_neg = df_neg.loc[df_neg["label"].astype(int) == 0].copy()
    df_neg = _attach_group_metadata(df_neg, meta_path)

    # Drop negative-only columns that predate the ENN positive panel schema.
    drop_neg = sorted(set(df_neg.columns) - set(df_pos.columns))
    if drop_neg:
        print(f"Dropping negative-only columns for schema parity: {drop_neg}")
        df_neg = df_neg.drop(columns=drop_neg)

    missing_on_neg = sorted(set(df_pos.columns) - set(df_neg.columns))
    if missing_on_neg:
        raise AssertionError(
            "Negatives missing columns present on positives (run enrich?): "
            f"{missing_on_neg}"
        )

    if set(df_pos.columns) != set(df_neg.columns):
        only_pos = sorted(set(df_pos.columns) - set(df_neg.columns))
        only_neg = sorted(set(df_neg.columns) - set(df_pos.columns))
        raise AssertionError(
            "Column mismatch between positive and negative panels!\n"
            f"  only in positives ({len(only_pos)}): {only_pos}\n"
            f"  only in negatives ({len(only_neg)}): {only_neg}"
        )

    if not df_neg["rfam_acc"].astype(str).str.startswith("REFSEQ:").all():
        bad = df_neg.loc[
            ~df_neg["rfam_acc"].astype(str).str.startswith("REFSEQ:"), "rfam_acc"
        ].unique()[:5]
        raise AssertionError(
            f"Negative rfam_acc must be REFSEQ:{{assembly}}; bad={bad}"
        )
    n_neg_groups = int(df_neg["rfam_acc"].nunique())
    if n_neg_groups <= 1:
        raise AssertionError(f"degenerate negative groups: {n_neg_groups}")

    for col in ("seq_start", "seq_end"):
        df_pos[col] = df_pos[col].astype(int)
        df_neg[col] = df_neg[col].astype(int)

    n_pos, n_neg = len(df_pos), len(df_neg)
    if n_pos != 1198 or n_neg != 1198:
        raise AssertionError(f"Expected 1198/1198 pos/neg; got {n_pos}/{n_neg}")

    pos_keys = df_pos[JOIN_COLUMNS].astype(str).agg("|".join, axis=1)
    neg_keys = df_neg[JOIN_COLUMNS].astype(str).agg("|".join, axis=1)
    if pos_keys.duplicated().any() or neg_keys.duplicated().any():
        raise AssertionError("Duplicate join keys in positive or negative panel")
    if set(pos_keys) & set(neg_keys):
        raise AssertionError("Join-key overlap between positives and negatives")

    merged = pd.concat(
        [df_pos[list(df_pos.columns)], df_neg[list(df_pos.columns)]],
        ignore_index=True,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    print(
        f"Wrote {out_path} n={len(merged)} "
        f"(pos={n_pos}, neg={n_neg}, neg_groups={n_neg_groups})"
    )
    return out_path


def _build_parser() -> argparse.ArgumentParser:
    """Build the RUS fused-panel merge CLI parser."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--positives-fused", default=DEFAULT_POS_FUSED)
    p.add_argument("--negatives-fused", default=DEFAULT_NEG_FUSED)
    p.add_argument("--negatives-meta", default=DEFAULT_NEG_META)
    p.add_argument("--output-csv", default=DEFAULT_OUT)
    return p


def main() -> None:
    """Merge RUS negatives with ENN-panel positives into a dedicated fused CSV."""
    args = _build_parser().parse_args()
    merge_rus_fused_panel(
        positives_fused=args.positives_fused,
        negatives_fused=args.negatives_fused,
        negatives_meta=args.negatives_meta,
        output_csv=args.output_csv,
    )


if __name__ == "__main__":
    main()
