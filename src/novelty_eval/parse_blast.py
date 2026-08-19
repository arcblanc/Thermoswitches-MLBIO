"""Parse BLAST tabular output for novelty evaluation."""

from pathlib import Path

import pandas as pd

BLAST_COLUMNS = [
    "qseqid",
    "sseqid",
    "pident",
    "length",
    "mismatch",
    "gapopen",
    "qstart",
    "qend",
    "sstart",
    "send",
    "evalue",
    "bitscore",
]


def load_blast_hits(tsv_path: Path | str, evalue_max: float = 0.1) -> pd.DataFrame:
    """Load BLAST tabular hits and keep rows at or below evalue_max."""
    path = Path(tsv_path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=BLAST_COLUMNS)

    df = pd.read_csv(path, sep="\t", header=None, names=BLAST_COLUMNS)
    df = df[df["evalue"] <= evalue_max].copy()
    if df.empty:
        return df

    df["identity_frac"] = df["pident"] / 100.0
    df["alignment_length"] = df["length"]
    df["tool"] = "blastn"
    return df


def best_blast_hit_per_query(df: pd.DataFrame) -> pd.DataFrame:
    """Return the best BLAST hit per query by e-value then identity."""
    if df.empty:
        return pd.DataFrame(
            columns=[
                "record_id",
                "blast_target_id",
                "blast_evalue",
                "blast_bitscore",
                "blast_alignment_length",
                "blast_identity_pct",
                "blast_identity_frac",
            ]
        )

    ranked = df.sort_values(
        ["qseqid", "evalue", "identity_frac"], ascending=[True, True, False]
    )
    best = ranked.groupby("qseqid", as_index=False).first()
    return best.rename(
        columns={
            "qseqid": "record_id",
            "sseqid": "blast_target_id",
            "evalue": "blast_evalue",
            "bitscore": "blast_bitscore",
            "alignment_length": "blast_alignment_length",
            "pident": "blast_identity_pct",
            "identity_frac": "blast_identity_frac",
        }
    )[
        [
            "record_id",
            "blast_target_id",
            "blast_evalue",
            "blast_bitscore",
            "blast_alignment_length",
            "blast_identity_pct",
            "blast_identity_frac",
        ]
    ]
