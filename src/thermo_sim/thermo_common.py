import json
import math
import os
import time
import tracemalloc
import warnings
from dataclasses import dataclass
from functools import wraps
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
from scipy.optimize import OptimizeWarning, curve_fit

from data_engineering.cd_hit_sequence_similarity import JOIN_COLUMNS, parse_fasta_header
from data_engineering.paths import PROJECT_ROOT, resolve_path

BALANCED_DIR = "data/processed/balanced"
VIENNA_OUTPUT_DIR = "data/processed/viennarna"
NUPACK_OUTPUT_DIR = "data/processed/nupack"
PROTOTYPE_DIR = "data/processed/prototype"
DEFAULT_BALANCED_CSV = f"{BALANCED_DIR}/balanced_dataset.csv"
DEFAULT_BALANCED_FASTA = f"{BALANCED_DIR}/balanced_dataset.fasta"
DEFAULT_PROTOTYPE_CSV = f"{PROTOTYPE_DIR}/prototype_panel.csv"
DEFAULT_PROTOTYPE_FASTA = f"{PROTOTYPE_DIR}/prototype_panel.fasta"
DEFAULT_TEMP_MIN = 20
DEFAULT_TEMP_MAX = 70
DEFAULT_TEMP_STEP = 5
PROTOTYPE_TEMP_MIN = 10
PROTOTYPE_TEMP_MAX = 80
PROTOTYPE_TEMP_STEP = 2

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

SD_MOTIFS = ("AGGAGG", "AGGAG", "AGGA")
PROTOTYPE_PANEL_SPECS = [
    {
        "role": "canonical_positive",
        "rfam_acc": "RF01795",
        "rfam_id": "FourU",
    },
    {
        "role": "anomaly_cspA",
        "rfam_acc": "RF01766",
        "rfam_id": "cspA",
        "rfamseq_acc": "JPER01000003.1",
        "seq_start": 127157,
        "seq_end": 127581,
    },
    {
        "role": "short_negative",
        "rfam_acc": "RF01068",
        "rfam_id": "Guanidine-II",
    },
    {
        "role": "stress_512nt",
        "rfam_acc": "RF01766",
        "rfam_id": "cspA",
        "rfamseq_acc": "LNAM01000175.1",
        "seq_start": 44100,
        "seq_end": 44611,
    },
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


def _attach_sequences(df, fasta_path):
    sequences = load_sequences_from_fasta(fasta_path)
    df = df.copy()
    for column in JOIN_COLUMNS:
        if column in {"seq_start", "seq_end"}:
            df[column] = df[column].astype(int)
    df["sequence"] = df.apply(
        lambda row: sequences[(row["rfamseq_acc"], row["seq_start"], row["seq_end"])],
        axis=1,
    )
    df["seq_length"] = df["sequence"].str.len()
    return df


def load_balanced_dataset(csv_path=DEFAULT_BALANCED_CSV, fasta_path=DEFAULT_BALANCED_FASTA):
    return _attach_sequences(pd.read_csv(resolve_path(csv_path)), fasta_path)


def load_prototype_panel(
    csv_path=DEFAULT_PROTOTYPE_CSV,
    fasta_path=DEFAULT_PROTOTYPE_FASTA,
):
    return _attach_sequences(pd.read_csv(resolve_path(csv_path)), fasta_path)


def export_prototype_panel(
    balanced_csv=DEFAULT_BALANCED_CSV,
    balanced_fasta=DEFAULT_BALANCED_FASTA,
    output_csv=DEFAULT_PROTOTYPE_CSV,
    output_fasta=DEFAULT_PROTOTYPE_FASTA,
):
    dataset = load_balanced_dataset(balanced_csv, balanced_fasta)
    selected_rows = []

    for spec in PROTOTYPE_PANEL_SPECS:
        subset = dataset[dataset["rfam_acc"] == spec["rfam_acc"]]
        if spec.get("rfam_id"):
            subset = subset[subset["rfam_id"] == spec["rfam_id"]]
        if spec.get("rfamseq_acc"):
            subset = subset[subset["rfamseq_acc"] == spec["rfamseq_acc"]]
        if spec.get("seq_start") is not None:
            subset = subset[
                (subset["seq_start"] == spec["seq_start"])
                & (subset["seq_end"] == spec["seq_end"])
            ]
        if spec.get("max_len"):
            subset = subset[subset["seq_length"] <= spec["max_len"]]
        if spec["role"] == "stress_512nt":
            subset = subset.nlargest(1, "seq_length")
        elif spec["role"] == "canonical_positive":
            subset = subset.nsmallest(1, "seq_length")
        elif spec["role"] == "anomaly_cspA":
            subset = subset.nlargest(1, "seq_length")
        if subset.empty:
            raise ValueError(f"Could not locate prototype panel member: {spec['role']}")
        row = subset.iloc[0].copy()
        row["panel_role"] = spec["role"]
        selected_rows.append(row)

    panel = pd.DataFrame(selected_rows).reset_index(drop=True)
    output_csv = resolve_path(output_csv)
    output_fasta = resolve_path(output_fasta)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    panel[METADATA_COLUMNS + ["panel_role", "seq_length"]].to_csv(output_csv, index=False)
    with output_fasta.open("w") as handle:
        for _, row in panel.iterrows():
            header = (
                f">{row['rfam_acc']}|{row['rfam_id']}|{row['rfamseq_acc']}|"
                f"{row['seq_start']}-{row['seq_end']}|label={int(row['label'])}|role={row['panel_role']}"
            )
            handle.write(header + "\n")
            handle.write(row["sequence"] + "\n")

    return panel


def detect_shine_dalgarno(sequence):
    """Return 0-based inclusive (start, end) indices for the SD motif or 3' window."""
    sequence = normalize_sequence(sequence)
    for motif in SD_MOTIFS:
        index = sequence.find(motif)
        if index >= 0:
            return index, index + len(motif) - 1

    window = min(10, len(sequence))
    return len(sequence) - window, len(sequence) - 1


def sd_window_indices(sequence):
    start, end = detect_shine_dalgarno(sequence)
    return list(range(start, end + 1))


def extract_pair_probability(matrix, i, j):
    """Read pair probability from Vienna bpp rows or a square numpy matrix."""
    if matrix is None:
        return None
    if isinstance(matrix, np.ndarray):
        if i >= matrix.shape[0] or j >= matrix.shape[1]:
            return None
        return float(matrix[i, j])
    if i < len(matrix) and j < len(matrix[i]):
        return float(matrix[i][j])
    if j < len(matrix) and i < len(matrix[j]):
        return float(matrix[j][i])
    return None


def mean_unpaired_from_pair_matrix(matrix, n):
    unpaired = []
    if isinstance(matrix, np.ndarray):
        for i in range(n):
            paired = float(matrix[i, :].sum() + matrix[:, i].sum() - matrix[i, i])
            unpaired.append(max(0.0, min(1.0, 1.0 - paired)))
        return unpaired

    for i in range(n):
        paired = 0.0
        row = matrix[i] if i < len(matrix) else ()
        for j in range(len(row)):
            if i != j:
                paired += float(row[j])
        for k in range(i):
            row_k = matrix[k]
            if i < len(row_k):
                paired += float(row_k[i])
        unpaired.append(max(0.0, min(1.0, 1.0 - paired)))
    return unpaired


def window_mean(values, indices):
    if not values or not indices:
        return None
    picked = [values[i] for i in indices if 0 <= i < len(values)]
    return float(sum(picked) / len(picked)) if picked else None


def hill_sigmoid(temps, bottom, top, tm, hill_coeff):
    temps = np.asarray(temps, dtype=float)
    safe_t = np.clip(temps, 1e-6, None)
    safe_tm = max(float(tm), 1e-6)
    safe_n = max(float(hill_coeff), 1e-6)
    return bottom + (top - bottom) / (1.0 + np.power(safe_tm / safe_t, safe_n))


def fit_hill_curve(temps, values):
    if not temps or not values or len(temps) != len(values):
        return _failed_hill_fit("invalid_input")

    temps_arr = np.asarray(temps, dtype=float)
    values_arr = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(values_arr)):
        return _failed_hill_fit("non_finite_values")

    bottom_obs = float(np.min(values_arr))
    top_obs = float(np.max(values_arr))
    amplitude_obs = top_obs - bottom_obs
    if amplitude_obs < 1e-6:
        return {
            "Tm": None,
            "hill_coeff": None,
            "amplitude": amplitude_obs,
            "bottom": bottom_obs,
            "top": top_obs,
            "fit_status": "flat",
            "rmse": 0.0,
        }

    mid_index = len(values_arr) // 2
    p0 = [
        bottom_obs,
        top_obs,
        float(temps_arr[mid_index]),
        2.0,
    ]
    lower = [min(bottom_obs, top_obs) - 0.5, min(bottom_obs, top_obs), 5.0, 0.1]
    upper = [max(bottom_obs, top_obs) + 0.5, max(bottom_obs, top_obs) + 0.5, 95.0, 20.0]

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", OptimizeWarning)
            params, _ = curve_fit(
                hill_sigmoid,
                temps_arr,
                values_arr,
                p0=p0,
                bounds=(lower, upper),
                maxfev=10_000,
            )
        fitted = hill_sigmoid(temps_arr, *params)
        rmse = float(np.sqrt(np.mean((fitted - values_arr) ** 2)))
        bottom, top, tm, hill_coeff = [float(x) for x in params]
        return {
            "Tm": tm,
            "hill_coeff": hill_coeff,
            "amplitude": top - bottom,
            "bottom": bottom,
            "top": top,
            "fit_status": "ok",
            "rmse": rmse,
        }
    except (RuntimeError, ValueError, OptimizeWarning):
        return _failed_hill_fit("optimize_failed")


def _failed_hill_fit(reason):
    return {
        "Tm": None,
        "hill_coeff": None,
        "amplitude": None,
        "bottom": None,
        "top": None,
        "fit_status": reason,
        "rmse": None,
    }


def _feature_table_columns(feature_columns, extra_columns=None, join_columns=None, include_label=True):
    join_columns = join_columns or JOIN_COLUMNS
    columns = list(join_columns)
    if include_label:
        columns.append("label")
    if extra_columns:
        columns.extend(extra_columns)
    columns.extend(feature_columns)
    return list(dict.fromkeys(columns))


def append_feature_table(
    df, output_path, feature_columns, extra_columns=None, join_columns=None, include_label=True
):
    output_path = resolve_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = _feature_table_columns(
        feature_columns, extra_columns, join_columns=join_columns, include_label=include_label
    )
    write_header = not output_path.exists() or output_path.stat().st_size == 0
    df[columns].to_csv(output_path, mode="a", header=write_header, index=False)
    return output_path


def write_feature_table(
    df, output_path, feature_columns, extra_columns=None, join_columns=None, include_label=True
):
    output_path = resolve_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = _feature_table_columns(
        feature_columns, extra_columns, join_columns=join_columns, include_label=include_label
    )
    df[columns].to_csv(output_path, index=False)
    return output_path


def write_json(path, payload):
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, default=str)
    return path


def peak_memory_mb(callable_obj, *args, **kwargs):
    process = psutil.Process(os.getpid())
    rss_before = process.memory_info().rss
    tracemalloc.start()
    try:
        result = callable_obj(*args, **kwargs)
    finally:
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    rss_after = process.memory_info().rss
    return {
        "result": result,
        "peak_traced_mb": peak / (1024 * 1024),
        "rss_delta_mb": (rss_after - rss_before) / (1024 * 1024),
        "rss_after_mb": rss_after / (1024 * 1024),
    }


def elapsed_seconds(callable_obj, *args, **kwargs):
    start = time.perf_counter()
    result = callable_obj(*args, **kwargs)
    return {"result": result, "elapsed_sec": time.perf_counter() - start}


def max_stem_length(dot_bracket):
    longest = 0
    current = 0
    for char in dot_bracket:
        if char == "(":
            current += 1
            longest = max(longest, current)
        elif char == ")":
            current = max(current - 1, 0)
    return longest


def max_loop_length(dot_bracket):
    longest = 0
    current = 0
    for char in dot_bracket:
        if char == ".":
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def gc_content(sequence):
    sequence = normalize_sequence(sequence)
    if not sequence:
        return 0.0
    gc = sum(1 for base in sequence if base in {"G", "C"})
    return gc / len(sequence)


FASTA_JOIN_COLUMNS = ["record_id"]
DEFAULT_DENOVO_FASTA = "data/processed/de_novo/generated.fasta"


def load_fasta_dataset(fasta_path=DEFAULT_DENOVO_FASTA):
    fasta_path = resolve_path(fasta_path)
    records = []
    header = None
    sequence_parts = []

    with fasta_path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    seq = normalize_sequence("".join(sequence_parts))
                    record_id = header[1:].split()[0]
                    records.append(
                        {
                            "record_id": record_id,
                            "sequence": seq,
                            "seq_length": len(seq),
                        }
                    )
                header = line
                sequence_parts = []
            else:
                sequence_parts.append(line)

    if header is not None:
        seq = normalize_sequence("".join(sequence_parts))
        record_id = header[1:].split()[0]
        records.append(
            {
                "record_id": record_id,
                "sequence": seq,
                "seq_length": len(seq),
            }
        )

    return pd.DataFrame(records)


@dataclass
class LocalNupackWheel:
    root: Path

    def wheel_for_current_platform(self):
        import platform
        import sys

        py_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
        machine = platform.machine()
        if machine == "arm64":
            platform_tag = "macosx_11_0_arm64"
        elif machine == "x86_64":
            platform_tag = "macosx_12_0_x86_64"
        else:
            platform_tag = f"linux_{machine}"
        pattern = f"nupack-4.1.0.1-{py_tag}-{py_tag}-{platform_tag}.whl"
        matches = sorted(self.root.glob(f"package/{pattern}"))
        if not matches:
            matches = sorted(self.root.glob("package/nupack-*.whl"))
        return matches[0] if matches else None


def local_nupack_wheel_path():
    return LocalNupackWheel(PROJECT_ROOT / "nupack-4.1.0.1").wheel_for_current_platform()
