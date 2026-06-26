import argparse
import shutil
import subprocess
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CDHIT_OUTPUT_DIR = "data/processed/cdhitoutput"
STAGING_DIR = f"{CDHIT_OUTPUT_DIR}/cdhit_input"
JOIN_COLUMNS = ["rfamseq_acc", "seq_start", "seq_end"]
DEFAULT_IDENTITY = 0.8
DEFAULT_THREADS = 0


def require_cd_hit():
    """Ensure cd-hit-est is available on PATH."""
    if shutil.which("cd-hit-est") is None:
        raise EnvironmentError(
            "cd-hit-est not found on PATH. "
            "Create the conda env: conda env create -f environment.yml && conda activate thermoswitches-mlbio"
        )


def resolve_path(path):
    """A navigation helper. It takes any file path and ensures 
    it is an absolute path (starting precisely from the project's root folder) 
    so the script never gets lost looking for files."""
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def stage_inputs(fasta_src, csv_src, label):
    """A safety mechanism.Copy raw FASTA/CSV into processed staging; return staged paths."""
    fasta_src = resolve_path(fasta_src)
    csv_src = resolve_path(csv_src)
    staging_dir = resolve_path(STAGING_DIR)
    staging_dir.mkdir(parents=True, exist_ok=True)

    staged_fasta = staging_dir / f"{label}.fasta"
    staged_csv = staging_dir / f"{label}.csv"

    shutil.copy2(fasta_src, staged_fasta)
    shutil.copy2(csv_src, staged_csv)

    print(f"Staged copy: {fasta_src} -> {staged_fasta}")
    print(f"Staged copy: {csv_src} -> {staged_csv}")
    print(f"Original preserved in {fasta_src.parent}/")

    return staged_fasta, staged_csv


def consolidate_cdhit_outputs():
    """Move legacy CD-HIT artifacts from data/processed/ into cdhitoutput/."""
    processed_dir = resolve_path("data/processed")
    output_dir = resolve_path(CDHIT_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    patterns = [
        "positives_cdhit.fasta",
        "positives_cdhit.fasta.clstr",
        "positives_deduped.fasta",
        "positives_deduped.csv",
        "negatives_cdhit.fasta",
        "negatives_cdhit.fasta.clstr",
        "negatives_deduped.fasta",
        "negatives_deduped.csv",
    ]

    moved = 0
    for name in patterns:
        source = processed_dir / name
        destination = output_dir / name
        if source.exists() and not destination.exists():
            shutil.move(str(source), str(destination))
            print(f"Moved {source.name} -> {destination}")
            moved += 1

    legacy_staging = processed_dir / "cdhit_input"
    new_staging = output_dir / "cdhit_input"
    if legacy_staging.exists() and legacy_staging != new_staging:
        new_staging.mkdir(parents=True, exist_ok=True)
        for item in legacy_staging.iterdir():
            destination = new_staging / item.name
            if not destination.exists():
                shutil.move(str(item), str(destination))
                print(f"Moved {item.name} -> {destination}")
                moved += 1
        if not any(legacy_staging.iterdir()):
            legacy_staging.rmdir()

    if moved:
        print(f"Consolidated {moved} CD-HIT artifact(s) into {output_dir}")
    else:
        print(f"CD-HIT artifacts already consolidated in {output_dir}")

    return output_dir


def parse_fasta_header(header):
    """Acts as a translator. The FASTA sequences have long, 
    complicated name tags (e.g., >RF00114|sRNA|AB12345|10-50). 
    This function chops that text up and turns it into clean, distinct variables."""
    
    cleaned = header.strip()
    if cleaned.startswith(">"):
        cleaned = cleaned[1:]

    parts = cleaned.split("|")
    if len(parts) < 4:
        raise ValueError(f"Unexpected FASTA header format: {header}")

    coord_parts = parts[3].split("-")
    if len(coord_parts) != 2:
        raise ValueError(f"Could not parse coordinates from header: {header}")

    return {
        "rfam_acc": parts[0],
        "rfam_id": parts[1],
        "rfamseq_acc": parts[2],
        "seq_start": int(coord_parts[0]),
        "seq_end": int(coord_parts[1]),
        "header": cleaned,
    }


def parse_representatives_from_fasta(cdhit_fasta):
    """parse_representatives_from_fasta(cdhit_fasta)
    Reads the final FASTA file that CD-HIT spits out (the survivors). 
    It takes all those unique genetic sequences and organizes them into a clean pandas data table."""
    
    cdhit_fasta = resolve_path(cdhit_fasta)
    records = []
    header = None
    sequence_parts = []

    with cdhit_fasta.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    record = parse_fasta_header(header)
                    record["sequence"] = "".join(sequence_parts)
                    records.append(record)
                header = line
                sequence_parts = []
            else:
                sequence_parts.append(line)

    if header is not None:
        record = parse_fasta_header(header)
        record["sequence"] = "".join(sequence_parts)
        records.append(record)

    return pd.DataFrame(records)


def parse_cluster_file(clstr_path):
    """Reads the .clstr file, which is basically CD-HIT's receipt. 
    It calculates the statistics for your final report, 
    such as how many duplicate clusters were formed and the size of the largest cluster."""
    clstr_path = resolve_path(clstr_path)
    cluster_sizes = []
    current_size = 0

    with clstr_path.open() as handle:
        for line in handle:
            line = line.strip()
            if line.startswith(">Cluster"):
                if current_size:
                    cluster_sizes.append(current_size)
                current_size = 0
            elif line:
                current_size += 1

    if current_size:
        cluster_sizes.append(current_size)

    if not cluster_sizes:
        return {"clusters": 0, "max_cluster_size": 0, "singletons": 0}

    return {
        "clusters": len(cluster_sizes),
        "max_cluster_size": max(cluster_sizes),
        "singletons": sum(size == 1 for size in cluster_sizes),
    }

#Core Engine - CD-HIT 

def run_cd_hit_est(
    input_fasta,
    output_fasta,
    identity=DEFAULT_IDENTITY,
    threads=DEFAULT_THREADS,
):
    """Run cd-hit-est on a nucleotide FASTA file."""
    require_cd_hit()

    input_fasta = resolve_path(input_fasta)
    output_fasta = resolve_path(output_fasta)
    output_fasta.parent.mkdir(parents=True, exist_ok=True)

    command = [
        "cd-hit-est",
        "-i",
        str(input_fasta),
        "-o",
        str(output_fasta),
        "-c",
        str(identity),
        "-T",
        str(threads),
        "-M",
        "0",
    ]
    print("Running:", " ".join(command))
    subprocess.run(command, check=True)
    return output_fasta


def relink_metadata(representatives_df, csv_df):
    """Inner-join CD-HIT survivors back to the original Rfam metadata CSV."""
    csv_df = csv_df.copy()
    representatives_df = representatives_df.copy()
    for column in JOIN_COLUMNS:
        if column in {"seq_start", "seq_end"}:
            csv_df[column] = csv_df[column].astype(int)
            representatives_df[column] = representatives_df[column].astype(int)

    rep_keys = representatives_df[JOIN_COLUMNS + ["sequence"]].copy()
    merged = pd.merge(rep_keys, csv_df, on=JOIN_COLUMNS, how="inner")

    if len(merged) != len(representatives_df):
        missing = len(representatives_df) - len(merged)
        raise ValueError(
            f"Metadata relink dropped {missing} representative sequence(s). "
            "Check FASTA header parsing against the CSV composite key."
        )

    return merged


def write_deduped_outputs(merged_df, output_fasta, output_csv):
    """Write deduplicated FASTA and CSV outputs with full metadata."""
    output_fasta = resolve_path(output_fasta)
    output_csv = resolve_path(output_csv)
    output_fasta.parent.mkdir(parents=True, exist_ok=True)

    metadata_columns = [
        "rfam_acc",
        "rfam_id",
        "description",
        "type",
        "rfamseq_acc",
        "seq_start",
        "seq_end",
        "tax_string",
    ]

    with output_fasta.open("w") as handle:
        for _, row in merged_df.iterrows():
            header = (
                f">{row['rfam_acc']}|{row['rfam_id']}|{row['rfamseq_acc']}|"
                f"{row['seq_start']}-{row['seq_end']}"
            )
            handle.write(header + "\n")
            handle.write(row["sequence"] + "\n")

    merged_df[metadata_columns].to_csv(output_csv, index=False)
    return output_fasta, output_csv


def _digest_cd_hit_pool(
    label,
    input_fasta,
    input_csv,
    output_fasta,
    output_csv,
    cdhit_fasta,
    identity=DEFAULT_IDENTITY,
    threads=DEFAULT_THREADS,
):
    """
    Stage inputs → Run CD-HIT → Parse results → Relink metadata → Write outputs → Print statistics.
    """
    input_fasta = resolve_path(input_fasta)
    input_csv = resolve_path(input_csv)
    output_fasta = resolve_path(output_fasta)
    output_csv = resolve_path(output_csv)
    cdhit_fasta = resolve_path(cdhit_fasta)

    staging_label = Path(input_fasta).stem
    staged_fasta, _staged_csv = stage_inputs(input_fasta, input_csv, staging_label)

    input_count = sum(1 for line in staged_fasta.open() if line.startswith(">"))
    csv_df = pd.read_csv(input_csv)
    if len(csv_df) != input_count:
        print(
            f"Warning: {label} FASTA records ({input_count}) "
            f"do not match CSV rows ({len(csv_df)})."
        )

    run_cd_hit_est(staged_fasta, cdhit_fasta, identity=identity, threads=threads)

    representatives_df = parse_representatives_from_fasta(cdhit_fasta)
    merged_df = relink_metadata(representatives_df, csv_df)
    write_deduped_outputs(merged_df, output_fasta, output_csv)

    cluster_stats = parse_cluster_file(f"{cdhit_fasta}.clstr")
    survivors = len(merged_df)
    removed = input_count - survivors

    print(f"{label} CD-HIT complete.")
    print(f"  input sequences: {input_count}")
    print(f"  survivors:       {survivors}")
    print(f"  removed:         {removed}")
    print(f"  clusters:        {cluster_stats['clusters']}")
    print(f"  singletons:      {cluster_stats['singletons']}")
    print(f"  max cluster:     {cluster_stats['max_cluster_size']}")
    print(f"  wrote FASTA:     {output_fasta}")
    print(f"  wrote CSV:       {output_csv}")

    return merged_df


def digest_positives_cd_hit(
    input_fasta="data/raw/positives.fasta",
    input_csv="data/raw/rfam_positives.csv",
    output_fasta=f"{CDHIT_OUTPUT_DIR}/positives_deduped.fasta",
    output_csv=f"{CDHIT_OUTPUT_DIR}/positives_deduped.csv",
    cdhit_fasta=f"{CDHIT_OUTPUT_DIR}/positives_cdhit.fasta",
    identity=DEFAULT_IDENTITY,
    threads=DEFAULT_THREADS,
):
    """Run CD-HIT on positive thermoswitches and relink metadata."""
    return _digest_cd_hit_pool(
        label="Positives",
        input_fasta=input_fasta,
        input_csv=input_csv,
        output_fasta=output_fasta,
        output_csv=output_csv,
        cdhit_fasta=cdhit_fasta,
        identity=identity,
        threads=threads,
    )


def digest_negatives_cd_hit(
    input_fasta="data/raw/negatives.fasta",
    input_csv="data/raw/rfam_negatives.csv",
    output_fasta=f"{CDHIT_OUTPUT_DIR}/negatives_deduped.fasta",
    output_csv=f"{CDHIT_OUTPUT_DIR}/negatives_deduped.csv",
    cdhit_fasta=f"{CDHIT_OUTPUT_DIR}/negatives_cdhit.fasta",
    identity=DEFAULT_IDENTITY,
    threads=DEFAULT_THREADS,
):
    """Run CD-HIT on negative controls and relink metadata."""
    return _digest_cd_hit_pool(
        label="Negatives",
        input_fasta=input_fasta,
        input_csv=input_csv,
        output_fasta=output_fasta,
        output_csv=output_csv,
        cdhit_fasta=cdhit_fasta,
        identity=identity,
        threads=threads,
    )


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Run CD-HIT deduplication and relink Rfam metadata."
    )
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--positives", action="store_true", help="Deduplicate positives.")
    group.add_argument("--negatives", action="store_true", help="Deduplicate negatives.")
    group.add_argument("--all", action="store_true", help="Deduplicate positives then negatives.")
    parser.add_argument(
        "--migrate",
        action="store_true",
        help="Move legacy CD-HIT artifacts into data/processed/cdhitoutput/.",
    )
    parser.add_argument(
        "-c",
        "--identity",
        type=float,
        default=DEFAULT_IDENTITY,
        help="CD-HIT sequence identity threshold (default: 0.8).",
    )
    parser.add_argument(
        "-T",
        "--threads",
        type=int,
        default=DEFAULT_THREADS,
        help="CD-HIT thread count (0 uses all cores).",
    )
    return parser


#Main Function

def main():
    args = _build_parser().parse_args()
    kwargs = {"identity": args.identity, "threads": args.threads}

    if args.migrate:
        consolidate_cdhit_outputs()
        if not (args.positives or args.negatives or args.all):
            return

    if not (args.positives or args.negatives or args.all or args.migrate):
        _build_parser().error(
            "one of --positives, --negatives, --all, or --migrate is required"
        )

    if args.all:
        digest_positives_cd_hit(**kwargs)
        digest_negatives_cd_hit(**kwargs)
    elif args.positives:
        digest_positives_cd_hit(**kwargs)
    elif args.negatives:
        digest_negatives_cd_hit(**kwargs)


if __name__ == "__main__":
    main()
    

