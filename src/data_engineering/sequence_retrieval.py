import argparse
from typing import cast
import csv
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
from Bio import Entrez
from dotenv import load_dotenv

SRC_ROOT = Path(__file__).resolve().parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data_engineering.paths import PROJECT_ROOT

load_dotenv(PROJECT_ROOT / ".env")

DEFAULT_SLEEP_SEC = 0.11
FAILURE_COLUMNS = [
    "index",
    "rfam_acc",
    "rfam_id",
    "rfamseq_acc",
    "seq_start",
    "seq_end",
    "error",
]


def configure_entrez() -> None:
    """Load NCBI credentials from environment and configure Entrez client."""
    email = os.environ.get("EMAIL")
    api_key = os.environ.get("NCBI_API_KEY")
    if not email:
        raise EnvironmentError("EMAIL is not set. Add it to your .env file.")
    if not api_key:
        raise EnvironmentError("NCBI_API_KEY is not set. Add it to your .env file.")

    setattr(Entrez, "email", email)
    setattr(Entrez, "api_key", api_key)
    setattr(Entrez, "tool", "Thermoswitch_Classifier")
    setattr(Entrez, "max_tries", 5)
    setattr(Entrez, "sleep_between_tries", 15)


def _read_checkpoint(checkpoint_path: Path) -> int:
    """Return the last completed row index from a checkpoint file, or -1."""
    if not checkpoint_path.exists():
        return -1
    with checkpoint_path.open() as handle:
        data = json.load(handle)
    return int(data.get("last_index", -1))


def _write_checkpoint(checkpoint_path: Path, index: int) -> None:
    """Persist the last successfully fetched row index."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with checkpoint_path.open("w") as handle:
        json.dump({"last_index": index}, handle)


def _log_failure(
    failures_path: Path, row: pd.Series, index: int, error: BaseException
) -> None:
    """Append a failed Entrez fetch to the failures CSV."""
    failures_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not failures_path.exists()
    with failures_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FAILURE_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "index": index,
                "rfam_acc": row["rfam_acc"],
                "rfam_id": row["rfam_id"],
                "rfamseq_acc": row["rfamseq_acc"],
                "seq_start": row["seq_start"],
                "seq_end": row["seq_end"],
                "error": str(error),
            }
        )


def _parse_fasta_sequence(fasta_text: str) -> str:
    """Extract concatenated sequence lines from an Entrez FASTA payload."""
    lines = [line.strip() for line in fasta_text.strip().splitlines() if line.strip()]
    if not lines:
        raise ValueError("Empty FASTA response from Entrez")
    sequence_lines = [line for line in lines if not line.startswith(">")]
    if not sequence_lines:
        raise ValueError("FASTA response contained no sequence lines")
    return "".join(sequence_lines)


def _build_header(row: pd.Series, index: int) -> str:
    """Build a FASTA header from Rfam coordinates and the row index."""
    return (
        f">{row['rfam_acc']}|{row['rfam_id']}|{row['rfamseq_acc']}|"
        f"{row['seq_start']}-{row['seq_end']}|idx={index}"
    )


def fetch_fasta_from_df(
    df: pd.DataFrame,
    output_fasta: str | Path,
    checkpoint_path: str | Path,
    failures_path: str | Path,
    fresh: bool = False,
    limit: int | None = None,
    sleep_sec: float = DEFAULT_SLEEP_SEC,
) -> int:
    """Fetch coordinate-sliced nucleotide sequences and append them to a FASTA file."""
    configure_entrez()

    output_fasta = Path(output_fasta)
    checkpoint_path = Path(checkpoint_path)
    failures_path = Path(failures_path)
    output_fasta.parent.mkdir(parents=True, exist_ok=True)

    if fresh:
        if output_fasta.exists():
            output_fasta.unlink()
        if checkpoint_path.exists():
            checkpoint_path.unlink()
        if failures_path.exists():
            failures_path.unlink()

    last_index = _read_checkpoint(checkpoint_path)
    resuming = last_index >= 0 and output_fasta.exists()
    file_mode = "a" if resuming else "w"

    if limit is not None:
        df = df.head(limit)

    total = len(df)
    print(f"Starting extraction of {total} sequences to {output_fasta}...")
    if resuming:
        print(f"Resuming after index {last_index}.")

    fetched = 0
    skipped = 0

    with output_fasta.open(file_mode) as fasta_file:
        for row_index, row in df.iterrows():
            index = int(cast(int, row_index))
            if index <= last_index:
                skipped += 1
                continue

            accession = row["rfamseq_acc"]
            seq_start = int(min(row["seq_start"], row["seq_end"]))
            seq_stop = int(max(row["seq_start"], row["seq_end"]))

            try:
                handle = Entrez.efetch(
                    db="nucleotide",
                    id=accession,
                    seq_start=seq_start,
                    seq_stop=seq_stop,
                    rettype="fasta",
                    retmode="text",
                )
                fasta_data = handle.read()
                handle.close()

                sequence = _parse_fasta_sequence(fasta_data)
                fasta_file.write(_build_header(row, index) + "\n")
                fasta_file.write(sequence + "\n")
                fasta_file.flush()

                _write_checkpoint(checkpoint_path, index)
                fetched += 1
            except Exception as error:
                print(f"Failed to fetch {accession} at index {index}: {error}")
                _log_failure(failures_path, row, index, error)

            time.sleep(sleep_sec)

            if fetched and fetched % 1000 == 0:
                print(f"Successfully fetched {fetched} sequences...")

    print(
        f"Finished {output_fasta.name}: fetched={fetched}, skipped={skipped}, total={total}"
    )
    return fetched


def fetch_positive_fasta(
    csv_path: str | Path = "data/raw/rfam_positives.csv",
    output_fasta: str | Path = "data/raw/positives.fasta",
    fresh: bool = False,
    limit: int | None = None,
) -> int:
    """Fetch positive thermoswitch sequences from NCBI Entrez."""
    df = pd.read_csv(csv_path)
    return fetch_fasta_from_df(
        df=df,
        output_fasta=output_fasta,
        checkpoint_path="data/raw/.fetch_checkpoint_positives.json",
        failures_path="data/raw/fetch_failures_positives.csv",
        fresh=fresh,
        limit=limit,
    )


def fetch_negative_fasta(
    csv_path: str | Path = "data/raw/rfam_negatives.csv",
    output_fasta: str | Path = "data/raw/negatives.fasta",
    fresh: bool = False,
    limit: int | None = None,
) -> int:
    """Fetch negative-control sequences from NCBI Entrez."""
    df = pd.read_csv(csv_path)
    return fetch_fasta_from_df(
        df=df,
        output_fasta=output_fasta,
        checkpoint_path="data/raw/.fetch_checkpoint_negatives.json",
        failures_path="data/raw/fetch_failures_negatives.csv",
        fresh=fresh,
        limit=limit,
    )


def _build_parser() -> argparse.ArgumentParser:
    """Build the Entrez FASTA retrieval CLI parser."""
    parser = argparse.ArgumentParser(
        description="Fetch Rfam coordinate slices from NCBI Entrez as FASTA files."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--positives", action="store_true", help="Fetch positive thermoswitches."
    )
    group.add_argument(
        "--negatives", action="store_true", help="Fetch negative controls."
    )
    group.add_argument(
        "--all", action="store_true", help="Fetch positives then negatives."
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore checkpoints and overwrite output files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Fetch only the first N rows (useful for smoke tests).",
    )
    return parser


def main() -> None:
    """Parse CLI arguments and fetch the selected FASTA pool."""
    args = _build_parser().parse_args()

    if args.all:
        fetch_positive_fasta(fresh=args.fresh, limit=args.limit)
        fetch_negative_fasta(fresh=args.fresh, limit=args.limit)
    elif args.positives:
        fetch_positive_fasta(fresh=args.fresh, limit=args.limit)
    elif args.negatives:
        fetch_negative_fasta(fresh=args.fresh, limit=args.limit)


if __name__ == "__main__":
    main()
