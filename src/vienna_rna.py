import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from thermo_common import (
    DEFAULT_BALANCED_CSV,
    DEFAULT_BALANCED_FASTA,
    DEFAULT_TEMP_MAX,
    DEFAULT_TEMP_MIN,
    DEFAULT_TEMP_STEP,
    VIENNA_OUTPUT_DIR,
    build_temp_range,
    fit_hill_curve,
    load_balanced_dataset,
    resolve_path,
    write_feature_table,
)

VIENNA_FEATURE_COLUMNS = [
    "viennarna_Tm",
    "viennarna_hill_coeff",
    "viennarna_amplitude",
    "viennarna_mean_unpaired_prob",
]


def require_vienna_rna():
    """Ensure ViennaRNA Python bindings are importable."""
    try:
        import RNA  # noqa: F401
    except ImportError as exc:
        raise EnvironmentError(
            "ViennaRNA Python bindings not found. "
            "Install via bioconda: conda install -c bioconda viennarna "
            "or pip: pip install viennarna"
        ) from exc


def simulate_melting_curve(sequence, temp_range):
    """Compute melting-curve values via RNAheat-equivalent API.

    Stub: not yet implemented.
    """
    require_vienna_rna()
    raise NotImplementedError("RNAheat melting-curve simulation is not yet implemented.")


def compute_unpaired_profile(sequence):
    """Compute position-wise unpaired probabilities via RNAplfold.

    Stub: not yet implemented.
    """
    require_vienna_rna()
    raise NotImplementedError("RNAplfold unpaired-profile computation is not yet implemented.")


def extract_vienna_features(row, temp_range):
    """Derive ViennaRNA feature vector for a single sequence."""
    sequence = row["sequence"]
    melting_values = simulate_melting_curve(sequence, temp_range)
    unpaired_profile = compute_unpaired_profile(sequence)
    hill = fit_hill_curve(temp_range, melting_values)
    mean_unpaired = sum(unpaired_profile) / len(unpaired_profile) if unpaired_profile else None

    return {
        "viennarna_Tm": hill["Tm"],
        "viennarna_hill_coeff": hill["hill_coeff"],
        "viennarna_amplitude": hill["amplitude"],
        "viennarna_mean_unpaired_prob": mean_unpaired,
    }


def run_vienna_pipeline(
    input_csv=DEFAULT_BALANCED_CSV,
    input_fasta=DEFAULT_BALANCED_FASTA,
    output_csv=f"{VIENNA_OUTPUT_DIR}/features.csv",
    temp_min=DEFAULT_TEMP_MIN,
    temp_max=DEFAULT_TEMP_MAX,
    temp_step=DEFAULT_TEMP_STEP,
    dry_run=True,
):
    dataset = load_balanced_dataset(input_csv, input_fasta)
    temp_range = build_temp_range(temp_min, temp_max, temp_step)
    output_path = resolve_path(output_csv)

    if dry_run:
        print(f"ViennaRNA dry-run: {len(dataset)} sequences")
        print(f"  input CSV:   {resolve_path(input_csv)}")
        print(f"  input FASTA: {resolve_path(input_fasta)}")
        print(f"  output:      {output_path}")
        print(f"  temp range:  {temp_range[0]}–{temp_range[-1]}°C (step {temp_step})")
        print(f"  features:    {', '.join(VIENNA_FEATURE_COLUMNS)}")
        print("  engines:     RNAheat (melting curve), RNAplfold (unpaired sub-states)")
        return dataset

    require_vienna_rna()
    feature_rows = []
    for _, row in dataset.iterrows():
        features = extract_vienna_features(row, temp_range)
        feature_rows.append({**row.to_dict(), **features})

    result = write_feature_table(
        pd.DataFrame(feature_rows),
        output_path,
        VIENNA_FEATURE_COLUMNS,
    )
    print(f"Wrote ViennaRNA features: {result}")
    return result


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Extract ViennaRNA thermodynamic features (RNAheat + RNAplfold)."
    )
    parser.add_argument("--input-csv", default=DEFAULT_BALANCED_CSV)
    parser.add_argument("--input-fasta", default=DEFAULT_BALANCED_FASTA)
    parser.add_argument("--output-csv", default=f"{VIENNA_OUTPUT_DIR}/features.csv")
    parser.add_argument("--temp-min", type=int, default=DEFAULT_TEMP_MIN)
    parser.add_argument("--temp-max", type=int, default=DEFAULT_TEMP_MAX)
    parser.add_argument("--temp-step", type=int, default=DEFAULT_TEMP_STEP)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Validate inputs and print planned feature schema without running engines.",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Execute ViennaRNA simulations (requires viennarna installed).",
    )
    return parser


def main():
    args = _build_parser().parse_args()
    run_vienna_pipeline(
        input_csv=args.input_csv,
        input_fasta=args.input_fasta,
        output_csv=args.output_csv,
        temp_min=args.temp_min,
        temp_max=args.temp_max,
        temp_step=args.temp_step,
        dry_run=not args.run,
    )


if __name__ == "__main__":
    main()
