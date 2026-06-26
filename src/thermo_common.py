import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cd_hit_sequence_similarity import JOIN_COLUMNS, parse_fasta_header, resolve_path

BALANCED_DIR = "data/processed/balanced"
VIENNA_OUTPUT_DIR = "data/processed/viennarna"
NUPACK_OUTPUT_DIR = "data/processed/nupack"
DEFAULT_BALANCED_CSV = f"{BALANCED_DIR}/balanced_dataset.csv"
DEFAULT_BALANCED_FASTA = f"{BALANCED_DIR}/balanced_dataset.fasta"
DEFAULT_TEMP_MIN = 20
DEFAULT_TEMP_MAX = 70
DEFAULT_TEMP_STEP = 5

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


def normalize_sequence(sequence):
    return sequence.upper().replace("T", "U")


def build_temp_range(temp_min=DEFAULT_TEMP_MIN, temp_max=DEFAULT_TEMP_MAX, temp_step=DEFAULT_TEMP_STEP):
    return list(range(temp_min, temp_max + 1, temp_step))


def load_sequences_from_fasta(fasta_path):
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


def load_balanced_dataset(
    csv_path=DEFAULT_BALANCED_CSV,
    fasta_path=DEFAULT_BALANCED_FASTA,
):
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
    return df


def fit_hill_curve(temps, values):
    """Fit a Hill/logistic curve to melting or exposure data.

    Stub: returns placeholder values until scipy-based fitting is implemented.
    """
    if not temps or not values or len(temps) != len(values):
        return {"Tm": None, "hill_coeff": None, "amplitude": None}

    # TODO: implement logistic sigmoid fit (Tm, hill coefficient, amplitude)
    return {"Tm": None, "hill_coeff": None, "amplitude": None}


def write_feature_table(df, output_path, feature_columns):
    output_path = resolve_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = JOIN_COLUMNS + ["label"] + list(feature_columns)
    df[columns].to_csv(output_path, index=False)
    return output_path
