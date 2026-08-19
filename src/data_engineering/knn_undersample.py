import argparse
import pickle
import sys
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
from imblearn.under_sampling import EditedNearestNeighbours, RandomUnderSampler
from sklearn.feature_extraction.text import CountVectorizer

SRC_ROOT = Path(__file__).resolve().parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data_engineering.cd_hit_sequence_similarity import JOIN_COLUMNS, parse_fasta_header
from data_engineering.paths import resolve_path

CDHIT_OUTPUT_DIR = "data/processed/cdhitoutput"
BALANCED_DIR = "data/processed/balanced"
RANDOM_STATE = 42
METADATA_COLUMNS = [
    "rfam_acc",
    "rfam_id",
    "description",
    "type",
    "rfamseq_acc",
    "seq_start",
    "seq_end",
    "tax_string",
    "label",
]


def normalize_sequence(sequence: str) -> str:
    """Uppercase a sequence and convert DNA T bases to RNA U."""
    return sequence.upper().replace("T", "U")


def iter_sequence_kmer_tokens(
    sequence: str, ks: tuple[int, ...] = (2, 3)
) -> Iterator[str]:
    """Yield 2-mer and 3-mer tokens for a single sequence."""
    sequence = normalize_sequence(sequence)
    for k in ks:
        if len(sequence) < k:
            continue
        for index in range(len(sequence) - k + 1):
            yield sequence[index : index + k]


def build_kmer_corpus(sequences: list[str], ks: tuple[int, ...] = (2, 3)) -> list[str]:
    """Build space-joined k-mer token strings for each sequence."""
    return [" ".join(iter_sequence_kmer_tokens(sequence, ks)) for sequence in sequences]


def build_kmer_matrix(
    sequences: list[str], ks: tuple[int, ...] = (2, 3)
) -> tuple[object, CountVectorizer]:
    """Fit a k-mer count matrix and return it with the vectorizer."""
    corpus = build_kmer_corpus(sequences, ks=ks)
    vectorizer = CountVectorizer(token_pattern=r"\S+")
    matrix = vectorizer.fit_transform(corpus)
    return matrix, vectorizer


def load_sequences_from_fasta(
    fasta_path: str | Path,
) -> dict[tuple[str, int, int], str]:
    """Parse FASTA records keyed by rfamseq_acc and coordinates."""
    fasta_path = resolve_path(fasta_path)
    records = {}
    header = None
    sequence_parts = []

    with fasta_path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    parsed = parse_fasta_header(header)
                    key = tuple(parsed[column] for column in JOIN_COLUMNS)
                    records[key] = normalize_sequence("".join(sequence_parts))
                header = line
                sequence_parts = []
            else:
                sequence_parts.append(line)

    if header is not None:
        parsed = parse_fasta_header(header)
        key = tuple(parsed[column] for column in JOIN_COLUMNS)
        records[key] = normalize_sequence("".join(sequence_parts))

    return records


def load_labeled_pool(
    csv_path: str | Path, fasta_path: str | Path, label: int
) -> pd.DataFrame:
    """Load CSV metadata, attach FASTA sequences, and stamp a class label."""
    csv_path = resolve_path(csv_path)
    fasta_path = resolve_path(fasta_path)
    df = pd.read_csv(csv_path)
    sequences = load_sequences_from_fasta(fasta_path)

    df = df.copy()
    for column in JOIN_COLUMNS:
        if column in {"seq_start", "seq_end"}:
            df[column] = df[column].astype(int)

    df["sequence"] = df.apply(
        lambda row: sequences[(row["rfamseq_acc"], row["seq_start"], row["seq_end"])],
        axis=1,
    )
    df["label"] = label
    return df


def write_dataset(
    df: pd.DataFrame, output_csv: str | Path, output_fasta: str | Path
) -> tuple[Path, Path]:
    """Write labeled metadata CSV and FASTA files for a dataset."""
    output_csv = resolve_path(output_csv)
    output_fasta = resolve_path(output_fasta)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    df[METADATA_COLUMNS].to_csv(output_csv, index=False)
    with output_fasta.open("w") as handle:
        for _, row in df.iterrows():
            header = (
                f">{row['rfam_acc']}|{row['rfam_id']}|{row['rfamseq_acc']}|"
                f"{row['seq_start']}-{row['seq_end']}|label={int(row['label'])}"
            )
            handle.write(header + "\n")
            handle.write(row["sequence"] + "\n")

    return output_csv, output_fasta


def balance_with_enn_rus(
    positives_csv: str | Path = f"{CDHIT_OUTPUT_DIR}/positives_deduped.csv",
    positives_fasta: str | Path = f"{CDHIT_OUTPUT_DIR}/positives_deduped.fasta",
    negatives_csv: str | Path = f"{CDHIT_OUTPUT_DIR}/negatives_deduped.csv",
    negatives_fasta: str | Path = f"{CDHIT_OUTPUT_DIR}/negatives_deduped.fasta",
    enn_output_csv: str | Path = f"{BALANCED_DIR}/enn_cleaned.csv",
    enn_output_fasta: str | Path = f"{BALANCED_DIR}/enn_cleaned.fasta",
    balanced_output_csv: str | Path = f"{BALANCED_DIR}/balanced_dataset.csv",
    balanced_output_fasta: str | Path = f"{BALANCED_DIR}/balanced_dataset.fasta",
    vectorizer_output: str | Path = f"{BALANCED_DIR}/kmer_vectorizer.pkl",
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Clean majority k-mers with ENN, then random-undersample to class balance."""
    positives_df = load_labeled_pool(positives_csv, positives_fasta, label=1)
    negatives_df = load_labeled_pool(negatives_csv, negatives_fasta, label=0)
    dataset = pd.concat([positives_df, negatives_df], ignore_index=True)

    print(
        f"Step 1 input: {len(dataset)} samples "
        f"({len(positives_df)} positives, {len(negatives_df)} negatives)"
    )

    x_kmer, vectorizer = build_kmer_matrix(dataset["sequence"].tolist())
    y = dataset["label"].to_numpy()
    print(f"Step 1 k-mer matrix: {x_kmer.shape[0]} samples, {x_kmer.shape[1]} features")

    vectorizer_path = resolve_path(vectorizer_output)
    vectorizer_path.parent.mkdir(parents=True, exist_ok=True)
    with vectorizer_path.open("wb") as handle:
        pickle.dump(vectorizer, handle)

    x_kmer_dense = x_kmer.toarray()
    print(f"Step 1 dense array: {x_kmer_dense.shape} (contiguous memory for ENN)")

    enn = EditedNearestNeighbours(
        n_neighbors=3,
        kind_sel="all",
        sampling_strategy="majority",
        n_jobs=-1,
    )
    x_enn, y_enn = enn.fit_resample(x_kmer_dense, y)
    enn_df = dataset.iloc[enn.sample_indices_].reset_index(drop=True)

    pos_after_enn = int((enn_df["label"] == 1).sum())
    neg_after_enn = int((enn_df["label"] == 0).sum())
    print(
        f"Step 2 ENN: {len(negatives_df)} neg -> {neg_after_enn} neg "
        f"({len(negatives_df) - neg_after_enn} borderline removed); "
        f"positives held at {pos_after_enn}"
    )
    write_dataset(enn_df, enn_output_csv, enn_output_fasta)

    n_pos = pos_after_enn
    rus = RandomUnderSampler(
        sampling_strategy={0: n_pos, 1: n_pos},
        random_state=random_state,
    )
    x_bal, y_bal = rus.fit_resample(x_enn, y_enn)
    balanced_df = enn_df.iloc[rus.sample_indices_].reset_index(drop=True)

    pos_final = int((balanced_df["label"] == 1).sum())
    neg_final = int((balanced_df["label"] == 0).sum())
    print(
        f"Step 3 RUS: -> {pos_final} pos + {neg_final} neg "
        f"(random_state={random_state})"
    )
    write_dataset(balanced_df, balanced_output_csv, balanced_output_fasta)

    print(f"Wrote ENN cleaned:  {resolve_path(enn_output_csv)}")
    print(f"Wrote balanced set: {resolve_path(balanced_output_csv)}")
    return balanced_df


def _build_parser() -> argparse.ArgumentParser:
    """Build the ENN/RUS balancing CLI parser."""
    parser = argparse.ArgumentParser(
        description="Balance CD-HIT negatives to match positives via k-mer ENN and RUS."
    )
    parser.add_argument(
        "--positives-csv",
        default=f"{CDHIT_OUTPUT_DIR}/positives_deduped.csv",
    )
    parser.add_argument(
        "--positives-fasta",
        default=f"{CDHIT_OUTPUT_DIR}/positives_deduped.fasta",
    )
    parser.add_argument(
        "--negatives-csv",
        default=f"{CDHIT_OUTPUT_DIR}/negatives_deduped.csv",
    )
    parser.add_argument(
        "--negatives-fasta",
        default=f"{CDHIT_OUTPUT_DIR}/negatives_deduped.fasta",
    )
    return parser


def main() -> None:
    """Parse CLI arguments and write the balanced dataset."""
    args = _build_parser().parse_args()
    balance_with_enn_rus(
        positives_csv=args.positives_csv,
        positives_fasta=args.positives_fasta,
        negatives_csv=args.negatives_csv,
        negatives_fasta=args.negatives_fasta,
    )


if __name__ == "__main__":
    main()
