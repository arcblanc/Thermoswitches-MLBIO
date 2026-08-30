"""Merge BLAST/nhmmer hits, categorize novelty, write thesis-ready reports."""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

SRC_ROOT = Path(__file__).resolve().parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data_engineering.paths import resolve_path
from novelty_eval.parse_blast import best_blast_hit_per_query, load_blast_hits
from novelty_eval.parse_nhmmer import (
    best_nhmmer_hits_from_tbl,
    enrich_nhmmer_identity_for_best,
)

DEFAULT_CANDIDATES_CSV = "data/processed/denovo_top_candidates.csv"
DEFAULT_QUERY_FASTA = "data/processed/novelty/candidates_99.fasta"
DEFAULT_RFAM_FA = "data/reference/rfam/14.9/Rfam.fa"
DEFAULT_BLAST_TSV = "data/processed/novelty/blastn_hits.tsv"
DEFAULT_NHMMER_TBL = "data/processed/novelty/nhmmer_hits.tbl"
DEFAULT_REPORT_CSV = "data/processed/novelty/novelty_report.csv"
DEFAULT_SUMMARY_JSON = "data/processed/novelty/novelty_summary.json"
DEFAULT_BY_CATEGORY_CSV = "data/processed/novelty/novelty_by_category.csv"

IDENTITY_NEAR_THRESHOLD = 0.90
EVALUE_MAX = 0.1


def _categorize(identity_frac: float, has_hit: bool) -> str:
    """Map identity and hit presence to a novelty category label."""
    if not has_hit:
        return "no_hit"
    if identity_frac >= IDENTITY_NEAR_THRESHOLD:
        return "identical_near"
    return "remote_homolog"


def _negated_evalue(value: object) -> float:
    """Return negative e-value for descending sort keys (missing -> -1.0)."""
    if value is None:
        return -1.0
    if isinstance(value, (int, float)):
        return -float(value)
    return -float(str(value))


def _pick_combined_best(
    blast_row: pd.Series, nhmmer_row: pd.Series
) -> dict[str, object]:
    """Choose the better of BLAST and nhmmer hits by identity then e-value."""
    candidates = []
    if pd.notna(blast_row.get("blast_evalue")):
        candidates.append(
            {
                "best_tool": "blastn",
                "best_target_id": blast_row.get("blast_target_id"),
                "best_evalue": blast_row.get("blast_evalue"),
                "best_bitscore": blast_row.get("blast_bitscore"),
                "best_alignment_length": blast_row.get("blast_alignment_length"),
                "best_identity_frac": blast_row.get("blast_identity_frac"),
                "best_identity_pct": blast_row.get("blast_identity_pct"),
            }
        )
    if pd.notna(nhmmer_row.get("nhmmer_evalue")):
        candidates.append(
            {
                "best_tool": "nhmmer",
                "best_target_id": nhmmer_row.get("nhmmer_target_id"),
                "best_evalue": nhmmer_row.get("nhmmer_evalue"),
                "best_bitscore": nhmmer_row.get("nhmmer_bitscore"),
                "best_alignment_length": nhmmer_row.get("nhmmer_alignment_length"),
                "best_identity_frac": nhmmer_row.get("nhmmer_identity_frac"),
                "best_identity_pct": nhmmer_row.get("nhmmer_identity_pct"),
            }
        )

    if not candidates:
        return {
            "best_tool": None,
            "best_target_id": None,
            "best_evalue": None,
            "best_bitscore": None,
            "best_alignment_length": None,
            "best_identity_frac": None,
            "best_identity_pct": None,
        }

    return max(
        candidates,
        key=lambda row: (
            row["best_identity_frac"] if row["best_identity_frac"] is not None else -1,
            _negated_evalue(row["best_evalue"]),
        ),
    )


def build_novelty_report(
    candidates_csv: Path | str = DEFAULT_CANDIDATES_CSV,
    query_fasta: Path | str = DEFAULT_QUERY_FASTA,
    rfam_fasta: Path | str = DEFAULT_RFAM_FA,
    blast_tsv: Path | str = DEFAULT_BLAST_TSV,
    nhmmer_tbl: Path | str = DEFAULT_NHMMER_TBL,
    report_csv: Path | str = DEFAULT_REPORT_CSV,
    summary_json: Path | str = DEFAULT_SUMMARY_JSON,
    by_category_csv: Path | str = DEFAULT_BY_CATEGORY_CSV,
    evalue_max: float = EVALUE_MAX,
) -> tuple[Path, Path]:
    """Merge BLAST/nhmmer hits, categorize novelty, and write report files."""
    candidates_csv = resolve_path(candidates_csv)
    query_fasta = resolve_path(query_fasta)
    rfam_fasta = resolve_path(rfam_fasta)
    blast_tsv = resolve_path(blast_tsv)
    nhmmer_tbl = resolve_path(nhmmer_tbl)
    report_csv = resolve_path(report_csv)
    summary_json = resolve_path(summary_json)
    by_category_csv = resolve_path(by_category_csv)

    record_ids = pd.read_csv(candidates_csv)["record_id"].astype(str).tolist()

    blast_hits = load_blast_hits(blast_tsv, evalue_max=evalue_max)
    blast_best = best_blast_hit_per_query(blast_hits)

    nhmmer_best = best_nhmmer_hits_from_tbl(nhmmer_tbl, evalue_max=evalue_max)
    nhmmer_best = enrich_nhmmer_identity_for_best(nhmmer_best, query_fasta, rfam_fasta)

    base = pd.DataFrame({"record_id": record_ids})
    merged = base.merge(blast_best, on="record_id", how="left").merge(
        nhmmer_best, on="record_id", how="left"
    )

    combined_rows = []
    for _, row in merged.iterrows():
        combined_rows.append(_pick_combined_best(row, row))
    combined = pd.DataFrame(combined_rows)
    report = pd.concat([merged, combined], axis=1)

    report["has_hit"] = report["best_evalue"].notna()
    report["novelty_category"] = report.apply(
        lambda row: _categorize(
            row["best_identity_frac"] if pd.notna(row["best_identity_frac"]) else 0.0,
            bool(row["has_hit"]),
        ),
        axis=1,
    )

    report_csv.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(report_csv, index=False)

    counts = report["novelty_category"].value_counts().to_dict()
    total = len(report)
    summary = {
        "n_candidates": total,
        "evalue_threshold": evalue_max,
        "identity_near_threshold": IDENTITY_NEAR_THRESHOLD,
        "categories": {
            key: {
                "count": int(counts.get(key, 0)),
                "fraction": round(counts.get(key, 0) / total, 4) if total else 0.0,
            }
            for key in ("identical_near", "remote_homolog", "no_hit")
        },
        "tools": {
            "blastn_hits": int(blast_best["blast_target_id"].notna().sum()),
            "nhmmer_hits": int(nhmmer_best["nhmmer_target_id"].notna().sum()),
        },
    }
    summary_json.write_text(json.dumps(summary, indent=2))

    by_category = report[
        [
            "record_id",
            "novelty_category",
            "best_tool",
            "best_target_id",
            "best_identity_pct",
            "best_evalue",
        ]
    ].sort_values(["novelty_category", "best_identity_pct"], ascending=[True, False])
    by_category.to_csv(by_category_csv, index=False)

    print(f"Wrote {len(report)} rows to {report_csv}")
    print(f"Summary: {summary_json}")
    print(f"By category: {by_category_csv}")
    for key in ("identical_near", "remote_homolog", "no_hit"):
        print(
            f"  {key}: {summary['categories'][key]['count']} ({summary['categories'][key]['fraction']:.1%})"
        )
    return report_csv, summary_json


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for novelty report generation."""
    parser = argparse.ArgumentParser(
        description="Build novelty report from BLAST and nhmmer outputs."
    )
    parser.add_argument("--candidates-csv", default=DEFAULT_CANDIDATES_CSV)
    parser.add_argument("--blast-tsv", default=DEFAULT_BLAST_TSV)
    parser.add_argument("--nhmmer-tbl", default=DEFAULT_NHMMER_TBL)
    parser.add_argument("--query-fasta", default=DEFAULT_QUERY_FASTA)
    parser.add_argument("--rfam-fasta", default=DEFAULT_RFAM_FA)
    parser.add_argument("--report-csv", default=DEFAULT_REPORT_CSV)
    parser.add_argument("--summary-json", default=DEFAULT_SUMMARY_JSON)
    parser.add_argument("--by-category-csv", default=DEFAULT_BY_CATEGORY_CSV)
    parser.add_argument("--evalue-max", type=float, default=EVALUE_MAX)
    return parser


def main() -> None:
    """Build a novelty report from BLAST and nhmmer outputs."""
    args = _build_parser().parse_args()
    build_novelty_report(
        candidates_csv=args.candidates_csv,
        query_fasta=args.query_fasta,
        rfam_fasta=args.rfam_fasta,
        blast_tsv=args.blast_tsv,
        nhmmer_tbl=args.nhmmer_tbl,
        report_csv=args.report_csv,
        summary_json=args.summary_json,
        by_category_csv=args.by_category_csv,
        evalue_max=args.evalue_max,
    )


if __name__ == "__main__":
    main()
