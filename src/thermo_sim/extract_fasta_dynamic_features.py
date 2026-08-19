"""Compute Vienna dynamic features from a gated de novo / EVA FASTA.

Runs Z-score, ΔP_RBS, ΔΔG, ensemble diversity (Q), and mean positional entropy (S)
without NUPACK or full Hill melting curves.

Shuffle RNG seeds are derived from FASTA headers via
``rfamseq_acc=record_id`` → ``get_stable_seed(f"{record_id}|0|{len}")`` inside
``extract_dynamic_vienna_features``, so re-runs are deterministic.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

SRC_ROOT = Path(__file__).resolve().parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data_engineering.paths import resolve_path
from thermo_sim.vienna_rna import (
    VIENNA_DYNAMIC_FEATURE_COLUMNS,
    extract_dynamic_vienna_features,
    require_vienna_rna,
)

DEFAULT_FASTA = "data/processed/de_novo/generated.fasta"
DEFAULT_OUTPUT = "data/processed/eva_pilot/dynamic_features.csv"


def iter_fasta_records(fasta_path: Path) -> Iterator[tuple[str, str]]:
    """Yield (header, sequence) pairs from a FASTA file."""
    header = None
    parts: list[str] = []
    with fasta_path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(parts)
                header = line[1:].split()[0]
                parts = []
            else:
                parts.append(line)
    if header is not None:
        yield header, "".join(parts)


def _worker(payload: dict) -> dict:
    """Compute Vienna dynamic features for one FASTA record payload."""
    feats = extract_dynamic_vienna_features(
        payload,
        n_shuffles=payload["_n_shuffles"],
        dangles=payload.get("_dangles", 2),
    )
    out = {
        "record_id": payload["rfamseq_acc"],
        "sequence": payload["sequence"],
        "length_nt": len(payload["sequence"]),
    }
    for col in VIENNA_DYNAMIC_FEATURE_COLUMNS:
        out[col] = feats.get(col)
    out["viennarna_mfe_zscore_shuffle_mode"] = feats.get(
        "viennarna_mfe_zscore_shuffle_mode"
    )
    return out


def extract_from_fasta(
    *,
    fasta: str = DEFAULT_FASTA,
    output: str = DEFAULT_OUTPUT,
    n_shuffles: int = 100,
    workers: int = 4,
    dangles: int = 2,
) -> Path:
    """Compute Vienna dynamic features for every sequence in a FASTA file."""
    require_vienna_rna()
    fasta_path = resolve_path(fasta)
    out_path = resolve_path(output)
    if not fasta_path.exists():
        raise FileNotFoundError(f"FASTA not found: {fasta_path}")

    payloads = []
    for record_id, sequence in iter_fasta_records(fasta_path):
        seq = sequence.replace("T", "U").upper()
        payloads.append(
            {
                # Header drives get_stable_seed via join_key inside extract_dynamic_vienna_features
                "rfamseq_acc": record_id,
                "seq_start": 0,
                "seq_end": len(seq),
                "sequence": seq,
                "_n_shuffles": n_shuffles,
                "_dangles": dangles,
            }
        )

    if not payloads:
        raise ValueError(f"No sequences in {fasta_path}")

    rows: list[dict] = []
    workers = max(1, int(workers))
    print(
        f"Computing dynamic features for {len(payloads)} sequences "
        f"(workers={workers}, n_shuffles={n_shuffles})"
    )

    if workers == 1:
        for index, payload in enumerate(payloads, start=1):
            rows.append(_worker(payload))
            if index % 25 == 0 or index == len(payloads):
                print(f"  {index}/{len(payloads)}")
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_worker, p): i for i, p in enumerate(payloads)}
            done = 0
            for fut in as_completed(futures):
                rows.append(fut.result())
                done += 1
                if done % 25 == 0 or done == len(payloads):
                    print(f"  {done}/{len(payloads)}")

    # Preserve FASTA order
    by_id = {row["record_id"]: row for row in rows}
    ordered = [by_id[p["rfamseq_acc"]] for p in payloads]
    df = pd.DataFrame(ordered)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows → {out_path}")
    return out_path


def _build_parser() -> argparse.ArgumentParser:
    """Build the FASTA dynamic-feature extraction CLI parser."""
    parser = argparse.ArgumentParser(
        description="Vienna dynamic features (Z, ΔP_RBS, ΔΔG, Q, S) from FASTA."
    )
    parser.add_argument("--fasta", default=DEFAULT_FASTA)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--n-shuffles", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dangles", type=int, default=2)
    return parser


def main() -> None:
    """Extract Vienna dynamic features from a FASTA file."""
    args = _build_parser().parse_args()
    extract_from_fasta(
        fasta=args.fasta,
        output=args.output,
        n_shuffles=args.n_shuffles,
        workers=args.workers,
        dangles=args.dangles,
    )


if __name__ == "__main__":
    main()
