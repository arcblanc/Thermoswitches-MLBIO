"""Parse nhmmer --tblout for novelty evaluation."""

from pathlib import Path
from typing import TypedDict

import pandas as pd

NHMMER_COLUMNS = [
    "target_name",
    "target_accession",
    "query_name",
    "query_accession",
    "hmm_from",
    "hmm_to",
    "ali_from",
    "ali_to",
    "env_from",
    "env_to",
    "sq_len",
    "strand",
    "evalue",
    "score",
    "bias",
]


class NhmmerBestCandidate(TypedDict):
    record_id: str
    nhmmer_target_id: str
    nhmmer_evalue: float
    nhmmer_bitscore: float
    nhmmer_alignment_length: int


def _parse_tblout_line(line: str) -> dict[str, str] | None:
    """Parse one nhmmer tblout line into a column dict, or None if skipped."""
    if not line.strip() or line.startswith("#"):
        return None

    parts = line.split()
    if len(parts) < 15:
        return None

    row = dict(zip(NHMMER_COLUMNS, parts[:15]))
    row["description"] = " ".join(parts[15:])
    return row


def best_nhmmer_hits_from_tbl(
    tbl_path: Path | str, evalue_max: float = 0.1
) -> pd.DataFrame:
    """Stream tblout and return only the best hit per query (memory-safe)."""
    path = Path(tbl_path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(
            columns=[
                "record_id",
                "nhmmer_target_id",
                "nhmmer_evalue",
                "nhmmer_bitscore",
                "nhmmer_alignment_length",
                "nhmmer_identity_frac",
                "nhmmer_identity_pct",
            ]
        )

    best: dict[str, NhmmerBestCandidate] = {}
    with open(path) as handle:
        for line in handle:
            row = _parse_tblout_line(line)
            if row is None:
                continue
            try:
                evalue = float(row["evalue"])
                score = float(row["score"])
                ali_from = float(row["ali_from"])
                ali_to = float(row["ali_to"])
            except ValueError:
                continue
            if evalue > evalue_max:
                continue

            query = row["query_name"]
            alignment_length = abs(ali_to - ali_from) + 1
            candidate: NhmmerBestCandidate = {
                "record_id": query,
                "nhmmer_target_id": row["target_name"],
                "nhmmer_evalue": evalue,
                "nhmmer_bitscore": score,
                "nhmmer_alignment_length": int(alignment_length),
            }
            prev = best.get(query)
            if prev is None or (evalue, -score) < (
                prev["nhmmer_evalue"],
                -prev["nhmmer_bitscore"],
            ):
                best[query] = candidate

    if not best:
        return pd.DataFrame(
            columns=[
                "record_id",
                "nhmmer_target_id",
                "nhmmer_evalue",
                "nhmmer_bitscore",
                "nhmmer_alignment_length",
                "nhmmer_identity_frac",
                "nhmmer_identity_pct",
            ]
        )

    df = pd.DataFrame(best.values())
    return df.assign(nhmmer_identity_frac=0.0, nhmmer_identity_pct=0.0)


def load_nhmmer_hits(tbl_path: Path | str, evalue_max: float = 0.1) -> pd.DataFrame:
    """Load all nhmmer tblout hits at or below evalue_max."""
    path = Path(tbl_path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=NHMMER_COLUMNS + ["description"])

    rows = []
    with open(path) as handle:
        for line in handle:
            row = _parse_tblout_line(line)
            if row is not None:
                rows.append(row)

    if not rows:
        return pd.DataFrame(columns=NHMMER_COLUMNS + ["description"])

    df = pd.DataFrame(rows)
    for col in (
        "score",
        "evalue",
        "ali_from",
        "ali_to",
        "hmm_from",
        "hmm_to",
        "sq_len",
    ):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[df["evalue"] <= evalue_max].copy()
    if df.empty:
        return df

    df["alignment_length"] = (df["ali_to"] - df["ali_from"]).abs() + 1
    df["tool"] = "nhmmer"
    return df


def _load_query_sequences(fasta_path: Path | str) -> dict[str, str]:
    """Load FASTA records as a mapping from header id to sequence."""
    queries = {}
    header = None
    parts = []
    with open(fasta_path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    queries[header] = "".join(parts)
                header = line[1:].split()[0]
                parts = []
            else:
                parts.append(line)
        if header is not None:
            queries[header] = "".join(parts)
    return queries


def _fetch_rfam_sequences(
    fasta_path: Path | str, entry_ids: list[str]
) -> dict[str, str]:
    """Load only requested Rfam FASTA records (single pass)."""
    wanted = set(entry_ids)
    found = {}
    if not wanted:
        return found

    header = None
    parts = []
    with open(fasta_path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header in wanted:
                    found[header] = "".join(parts)
                    if len(found) == len(wanted):
                        break
                header = line[1:].split()[0]
                parts = []
            elif header in wanted:
                parts.append(line)
        if header in wanted and header not in found:
            found[header] = "".join(parts)
    return found


def _pairwise_identity(query_seq: str, target_seq: str) -> float:
    """Return global pairwise identity between query and target RNA."""
    from Bio.Align import PairwiseAligner

    query_seq = query_seq.upper().replace("T", "U")
    target_seq = target_seq.upper().replace("T", "U")
    if not query_seq or not target_seq:
        return 0.0

    aligner = PairwiseAligner(mode="global")
    aligner.match_score = 2
    aligner.mismatch_score = -1
    aligner.open_gap_score = -2
    aligner.extend_gap_score = -0.5
    alignment = aligner.align(query_seq, target_seq)[0]
    aligned_query = str(alignment[0])
    aligned_target = str(alignment[1])
    matches = sum(
        1
        for q, t in zip(aligned_query, aligned_target)
        if q == t and q != "-" and t != "-"
    )
    ali_len = sum(1 for q in aligned_query if q != "-")
    return matches / ali_len if ali_len else 0.0


def enrich_nhmmer_identity(
    tbl_df: pd.DataFrame,
    query_fasta: Path | str,
    rfam_fasta: Path | str,
    max_rows: int | None = None,
) -> pd.DataFrame:
    """Compute identity for nhmmer hits via FASTA fetch + pairwise alignment."""
    tbl_df = tbl_df.copy()
    if tbl_df.empty:
        tbl_df["identity_frac"] = []
        tbl_df["n_ident"] = []
        return tbl_df

    queries = _load_query_sequences(query_fasta)
    rows = tbl_df if max_rows is None else tbl_df.head(max_rows)
    target_ids = (
        rows["target_name"].tolist()
        if "target_name" in rows.columns
        else rows["nhmmer_target_id"].tolist()
    )
    targets = _fetch_rfam_sequences(rfam_fasta, target_ids)

    identity_values = []
    for _, row in rows.iterrows():
        query_name = (
            str(row["query_name"])
            if "query_name" in row.index
            else str(row["record_id"])
        )
        target_id = (
            str(row["target_name"])
            if "target_name" in row.index
            else str(row["nhmmer_target_id"])
        )
        query_seq = queries.get(query_name)
        target_seq = targets.get(target_id)
        if not query_seq or not target_seq:
            identity_values.append(0.0)
            continue
        identity_values.append(_pairwise_identity(query_seq, target_seq))

    if max_rows is not None:
        tbl_df = tbl_df.copy()
        tbl_df.loc[rows.index, "identity_frac"] = identity_values
        tbl_df["identity_frac"] = tbl_df["identity_frac"].fillna(0.0)
    else:
        tbl_df["identity_frac"] = identity_values

    tbl_df["n_ident"] = tbl_df["identity_frac"] * tbl_df["alignment_length"]
    return tbl_df


def enrich_nhmmer_identity_for_best(
    best_df: pd.DataFrame, query_fasta: Path | str, rfam_fasta: Path | str
) -> pd.DataFrame:
    """Fill identity columns for a best-hit-per-query nhmmer table."""
    if best_df.empty:
        return best_df
    enriched = enrich_nhmmer_identity(
        best_df.rename(
            columns={
                "nhmmer_target_id": "target_name",
                "record_id": "query_name",
                "nhmmer_alignment_length": "alignment_length",
            }
        ),
        query_fasta=query_fasta,
        rfam_fasta=rfam_fasta,
    )
    best_df = best_df.copy()
    best_df["nhmmer_identity_frac"] = enriched["identity_frac"].values
    best_df["nhmmer_identity_pct"] = best_df["nhmmer_identity_frac"] * 100.0
    return best_df


def best_nhmmer_hit_per_query(df: pd.DataFrame) -> pd.DataFrame:
    """Return the best nhmmer hit per query by e-value then score."""
    if df.empty:
        return pd.DataFrame(
            columns=[
                "record_id",
                "nhmmer_target_id",
                "nhmmer_evalue",
                "nhmmer_bitscore",
                "nhmmer_alignment_length",
                "nhmmer_identity_frac",
                "nhmmer_identity_pct",
            ]
        )

    ranked = df.sort_values(
        ["query_name", "evalue", "score"], ascending=[True, True, False]
    )
    best = ranked.groupby("query_name", as_index=False).first()
    return best.rename(
        columns={
            "query_name": "record_id",
            "target_name": "nhmmer_target_id",
            "evalue": "nhmmer_evalue",
            "score": "nhmmer_bitscore",
            "alignment_length": "nhmmer_alignment_length",
        }
    ).assign(nhmmer_identity_frac=0.0, nhmmer_identity_pct=0.0)[
        [
            "record_id",
            "nhmmer_target_id",
            "nhmmer_evalue",
            "nhmmer_bitscore",
            "nhmmer_alignment_length",
            "nhmmer_identity_frac",
            "nhmmer_identity_pct",
        ]
    ]
