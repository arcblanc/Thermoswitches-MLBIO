#!/usr/bin/env python3
"""Biophysical yield ratio for EVA / de novo gated FASTA.

Yield Ratio =
  #{ Z <= -2 AND ΔP_RBS > 0 AND E_Rfam > 1e-3 } / N_quality_gated

E_Rfam = min(best blastn e-value, best nhmmer e-value).
Sequences with zero hits in both searches get E_Rfam = +inf (pass novelty gate).
Missing IDs in hit tables never raise — left-join with default inf.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data_engineering.paths import resolve_path
from novelty_eval.parse_blast import best_blast_hit_per_query, load_blast_hits
from novelty_eval.parse_nhmmer import best_nhmmer_hits_from_tbl

DEFAULT_DYNAMIC = "data/processed/eva_pilot/dynamic_features.csv"
DEFAULT_BLAST = "data/processed/novelty/blastn_hits.tsv"
DEFAULT_NHMMER = "data/processed/novelty/nhmmer_hits.tbl"
DEFAULT_OUT_CSV = "data/processed/eva_pilot/yield_ratio_sequences.csv"
DEFAULT_OUT_JSON = "data/processed/eva_pilot/yield_ratio.json"

Z_MAX = -2.0
DELTA_P_MIN = 0.0
EVALUE_NOVELTY_MIN = 1e-3
# Load search hits up to this threshold (search itself often uses 0.1)
SEARCH_EVALUE_MAX = 0.1


def _to_finite_float(value: object) -> float | None:
    """Coerce a scalar to float when it is a finite numeric value."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            parsed = float(value)
        except ValueError:
            return None
        if math.isnan(parsed):
            return None
        return parsed
    return None


def _best_e_rfam(blast_e: object, nhmmer_e: object) -> float:
    """Return the minimum finite blast/nhmmer e-value, or +inf if none."""
    values: list[float] = []
    for raw in (blast_e, nhmmer_e):
        parsed = _to_finite_float(raw)
        if parsed is not None:
            values.append(parsed)
    if not values:
        return float("inf")
    return min(values)


def _gate_counts(
    merged: pd.DataFrame, z_max: float, delta_p_min: float, evalue_novelty_min: float
) -> dict[str, int]:
    """Count how many rows pass each individual yield-ratio gate."""
    z_ok = (merged["viennarna_mfe_zscore"].notna()) & (
        merged["viennarna_mfe_zscore"].astype(float) <= z_max
    )
    dp_ok = (merged["viennarna_delta_P_RBS"].notna()) & (
        merged["viennarna_delta_P_RBS"].astype(float) > delta_p_min
    )

    def _novel(e: object) -> bool:
        """Return True if E_Rfam is inf or strictly above the novelty threshold."""
        if isinstance(e, float) and math.isinf(e):
            return True
        parsed = _to_finite_float(e)
        return parsed is not None and parsed > evalue_novelty_min

    e_ok = merged["E_Rfam"].map(_novel)
    return {
        "z_le_m2": int(z_ok.sum()),
        "dp_gt_0": int(dp_ok.sum()),
        "novel": int(e_ok.sum()),
    }


def compute_yield_ratio(
    *,
    dynamic_csv: str = DEFAULT_DYNAMIC,
    blast_tsv: str = DEFAULT_BLAST,
    nhmmer_tbl: str = DEFAULT_NHMMER,
    output_csv: str = DEFAULT_OUT_CSV,
    output_json: str = DEFAULT_OUT_JSON,
    z_max: float = Z_MAX,
    delta_p_min: float = DELTA_P_MIN,
    evalue_novelty_min: float = EVALUE_NOVELTY_MIN,
    search_evalue_max: float = SEARCH_EVALUE_MAX,
    chunk_id: str | None = None,
    output_chunk_json: str | None = None,
    prior_chunks: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    """Compute yield ratio, write sequence CSV and summary JSON, and return it."""
    dynamic_path = resolve_path(dynamic_csv)
    dyn = pd.read_csv(dynamic_path)
    if "record_id" not in dyn.columns:
        raise ValueError(f"{dynamic_path} missing record_id column")

    record_ids = dyn["record_id"].astype(str).tolist()
    n_total = len(record_ids)

    blast_hits = load_blast_hits(resolve_path(blast_tsv), evalue_max=search_evalue_max)
    blast_best = best_blast_hit_per_query(blast_hits)
    nhmmer_best = best_nhmmer_hits_from_tbl(
        resolve_path(nhmmer_tbl), evalue_max=search_evalue_max
    )

    # Left-join: every gated ID present; missing hits → NaN → inf
    base = dyn.copy()
    base["record_id"] = base["record_id"].astype(str)
    merged = base.merge(blast_best, on="record_id", how="left").merge(
        nhmmer_best, on="record_id", how="left"
    )

    e_rfam = []
    passes = []
    for _, row in merged.iterrows():
        e_val = _best_e_rfam(row.get("blast_evalue"), row.get("nhmmer_evalue"))
        e_rfam.append(e_val)
        z = row.get("viennarna_mfe_zscore")
        dp = row.get("viennarna_delta_P_RBS")
        try:
            z_ok = pd.notna(z) and float(z) <= z_max
            dp_ok = pd.notna(dp) and float(dp) > delta_p_min
            e_ok = e_val > evalue_novelty_min
            passes.append(bool(z_ok and dp_ok and e_ok))
        except (TypeError, ValueError):
            passes.append(False)

    merged["E_Rfam"] = e_rfam
    merged["yield_pass"] = passes
    n_pass = int(sum(passes))
    n_no_hit = int(sum(math.isinf(e) for e in e_rfam))
    ratio = (n_pass / n_total) if n_total else 0.0

    out_csv = resolve_path(output_csv)
    out_json = resolve_path(output_json)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    # Serialize inf as string for CSV friendliness
    export = merged.copy()
    export["E_Rfam"] = [
        "inf" if math.isinf(e) else e for e in export["E_Rfam"].tolist()
    ]
    export.to_csv(out_csv, index=False)

    gate = _gate_counts(merged, z_max, delta_p_min, evalue_novelty_min)
    summary = {
        "n_quality_gated": n_total,
        "n_yield_pass": n_pass,
        "yield_ratio": ratio,
        "rolling_yield": ratio,
        "n_E_Rfam_inf_no_hit": n_no_hit,
        "z_le_m2": gate["z_le_m2"],
        "dp_gt_0": gate["dp_gt_0"],
        "novel": gate["novel"],
        "thresholds": {
            "viennarna_mfe_zscore_max": z_max,
            "viennarna_delta_P_RBS_min_exclusive": delta_p_min,
            "E_Rfam_gt": evalue_novelty_min,
        },
        "passing_record_ids": merged.loc[merged["yield_pass"], "record_id"].tolist(),
        "dynamic_csv": str(dynamic_path),
        "blast_tsv": str(resolve_path(blast_tsv)),
        "nhmmer_tbl": str(resolve_path(nhmmer_tbl)),
        "output_csv": str(out_csv),
        "chunks": list(prior_chunks or []),
    }

    if chunk_id:
        chunk_summary = {
            "chunk_id": chunk_id,
            "n": n_total,
            "n_pass": n_pass,
            "yield_ratio": ratio,
            "z_le_m2": gate["z_le_m2"],
            "dp_gt_0": gate["dp_gt_0"],
            "novel": gate["novel"],
        }
        if output_chunk_json:
            chunk_path = resolve_path(output_chunk_json)
            chunk_path.parent.mkdir(parents=True, exist_ok=True)
            chunk_path.write_text(json.dumps(chunk_summary, indent=2) + "\n")
            print(f"Wrote chunk summary {chunk_path}")
        summary["chunk_id"] = chunk_id
        summary["chunk"] = chunk_summary

    out_json.write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"Yield ratio: {n_pass}/{n_total} = {ratio:.4f} (no-hit/inf E_Rfam={n_no_hit})"
    )
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_json}")
    return summary


def _build_parser() -> argparse.ArgumentParser:
    """Build the yield-ratio argument parser."""
    parser = argparse.ArgumentParser(description="EVA biophysical yield ratio.")
    parser.add_argument("--dynamic-csv", default=DEFAULT_DYNAMIC)
    parser.add_argument("--blast-tsv", default=DEFAULT_BLAST)
    parser.add_argument("--nhmmer-tbl", default=DEFAULT_NHMMER)
    parser.add_argument("--output-csv", default=DEFAULT_OUT_CSV)
    parser.add_argument("--output-json", default=DEFAULT_OUT_JSON)
    parser.add_argument("--z-max", type=float, default=Z_MAX)
    parser.add_argument("--delta-p-min", type=float, default=DELTA_P_MIN)
    parser.add_argument("--evalue-novelty-min", type=float, default=EVALUE_NOVELTY_MIN)
    parser.add_argument("--chunk-id", default=None)
    parser.add_argument("--output-chunk-json", default=None)
    return parser


def main() -> None:
    """Parse CLI args and compute the EVA biophysical yield ratio."""
    args = _build_parser().parse_args()
    compute_yield_ratio(
        dynamic_csv=args.dynamic_csv,
        blast_tsv=args.blast_tsv,
        nhmmer_tbl=args.nhmmer_tbl,
        output_csv=args.output_csv,
        output_json=args.output_json,
        z_max=args.z_max,
        delta_p_min=args.delta_p_min,
        evalue_novelty_min=args.evalue_novelty_min,
        chunk_id=args.chunk_id,
        output_chunk_json=args.output_chunk_json,
    )


if __name__ == "__main__":
    main()
