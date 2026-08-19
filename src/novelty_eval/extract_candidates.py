"""Extract top de novo candidate sequences into a query FASTA for novelty search."""

import argparse
import sys
from collections.abc import Iterator
from pathlib import Path

import pandas as pd

SRC_ROOT = Path(__file__).resolve().parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data_engineering.paths import resolve_path

DEFAULT_CANDIDATES_CSV = "data/processed/denovo_top_candidates.csv"
DEFAULT_SOURCE_FASTA = "data/processed/de_novo/generated.fasta"
DEFAULT_OUTPUT_FASTA = "data/processed/novelty/candidates_99.fasta"


def _iter_fasta_records(fasta_path: Path | str) -> Iterator[tuple[str, str]]:
    """Yield (header, sequence) pairs from a FASTA file."""
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
    candidates_csv: Path | str = DEFAULT_CANDIDATES_CSV,
    source_fasta: Path | str = DEFAULT_SOURCE_FASTA,
    output_fasta: Path | str = DEFAULT_OUTPUT_FASTA,
    from_source_fasta: bool = False,
) -> Path:
    """Write a query FASTA of candidate IDs found in the source FASTA."""
    source_fasta = resolve_path(source_fasta)
    output_fasta = resolve_path(output_fasta)

    if from_source_fasta:
        ids = []
        found = {}
        for record_id, sequence in _iter_fasta_records(source_fasta):
            if record_id not in found:
                found[record_id] = sequence
                ids.append(record_id)
        if not ids:
            raise ValueError(f"No sequences in {source_fasta}")
        candidates_csv = resolve_path(candidates_csv)
        candidates_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"record_id": ids}).to_csv(candidates_csv, index=False)
        print(f"Wrote {len(ids)} IDs → {candidates_csv}")
    else:
        candidates_csv = resolve_path(candidates_csv)
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


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for candidate FASTA extraction."""
    parser = argparse.ArgumentParser(
        description="Extract novelty-search query FASTA from top candidates."
    )
    parser.add_argument("--candidates-csv", default=DEFAULT_CANDIDATES_CSV)
    parser.add_argument("--source-fasta", default=DEFAULT_SOURCE_FASTA)
    parser.add_argument("--output-fasta", default=DEFAULT_OUTPUT_FASTA)
    parser.add_argument(
        "--from-source-fasta",
        action="store_true",
        help="Use every record in --source-fasta (writes --candidates-csv from headers).",
    )
    return parser


def main() -> None:
    """Extract novelty-search query FASTA from the command line."""
    args = _build_parser().parse_args()
    extract_candidate_fasta(
        candidates_csv=args.candidates_csv,
        source_fasta=args.source_fasta,
        output_fasta=args.output_fasta,
        from_source_fasta=args.from_source_fasta,
    )


if __name__ == "__main__":
    main()
