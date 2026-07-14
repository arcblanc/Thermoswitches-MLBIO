"""Extract top de novo candidate sequences into a query FASTA for novelty search."""

import argparse
import sys
from pathlib import Path

import pandas as pd

SRC_ROOT = Path(__file__).resolve().parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data_engineering.paths import resolve_path

DEFAULT_CANDIDATES_CSV = "data/processed/denovo_top_candidates.csv"
DEFAULT_SOURCE_FASTA = "data/processed/de_novo/generated.fasta"
DEFAULT_OUTPUT_FASTA = "data/processed/novelty/candidates_99.fasta"


def _iter_fasta_records(fasta_path):
    header = None
    seq_parts = []
    with open(fasta_path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seq_parts)
                header = line[1:].split()[0]
                seq_parts = []
            else:
                seq_parts.append(line)
        if header is not None:
            yield header, "".join(seq_parts)


def extract_candidate_fasta(
    candidates_csv=DEFAULT_CANDIDATES_CSV,
    source_fasta=DEFAULT_SOURCE_FASTA,
    output_fasta=DEFAULT_OUTPUT_FASTA,
):
    candidates_csv = resolve_path(candidates_csv)
    source_fasta = resolve_path(source_fasta)
    output_fasta = resolve_path(output_fasta)

    ids = pd.read_csv(candidates_csv)["record_id"].astype(str).tolist()
    if not ids:
        raise ValueError(f"No record_id values in {candidates_csv}")

    wanted = set(ids)
    found = {}
    for record_id, sequence in _iter_fasta_records(source_fasta):
        if record_id in wanted and record_id not in found:
            found[record_id] = sequence

    missing = [record_id for record_id in ids if record_id not in found]
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(
            f"Missing {len(missing)} candidate(s) in {source_fasta}: {preview}"
            + (" ..." if len(missing) > 5 else "")
        )

    output_fasta.parent.mkdir(parents=True, exist_ok=True)
    with open(output_fasta, "w") as handle:
        for record_id in ids:
            handle.write(f">{record_id}\n{found[record_id]}\n")

    print(f"Wrote {len(ids)} sequences to {output_fasta}")
    return output_fasta


def _build_parser():
    parser = argparse.ArgumentParser(description="Extract novelty-search query FASTA from top candidates.")
    parser.add_argument("--candidates-csv", default=DEFAULT_CANDIDATES_CSV)
    parser.add_argument("--source-fasta", default=DEFAULT_SOURCE_FASTA)
    parser.add_argument("--output-fasta", default=DEFAULT_OUTPUT_FASTA)
    return parser


def main():
    args = _build_parser().parse_args()
    extract_candidate_fasta(
        candidates_csv=args.candidates_csv,
        source_fasta=args.source_fasta,
        output_fasta=args.output_fasta,
    )


if __name__ == "__main__":
    main()
