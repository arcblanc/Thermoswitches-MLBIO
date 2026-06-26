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
    NUPACK_OUTPUT_DIR,
    build_temp_range,
    fit_hill_curve,
    load_balanced_dataset,
    resolve_path,
    write_feature_table,
)

NUPACK_FEATURE_COLUMNS = [
    "nupack_Tm",
    "nupack_hill_coeff",
    "nupack_amplitude",
    "nupack_mean_exposure",
]

DEFAULT_STRAND_CONCENTRATION = 1e-8


def require_nupack():
    """Ensure NUPACK 4 Python module is importable."""
    try:
        from nupack import Model, SetSpec, Strand, Tube, tube_analysis  # noqa: F401
    except ImportError as exc:
        raise EnvironmentError(
            "NUPACK Python module not found. "
            "Install via pip: pip install nupack "
            "(requires an active paid NUPACK subscription)."
        ) from exc


def build_single_strand_tube(sequence, concentration=DEFAULT_STRAND_CONCENTRATION):
    """Construct a single-strand NUPACK test tube for ensemble analysis."""
    from nupack import SetSpec, Strand, Tube

    strand = Strand(sequence, name="utr")
    return Tube(
        strands={strand: concentration},
        complexes=SetSpec(max_size=1),
        name="single_strand",
    )


def simulate_exposure_curve(sequence, temp_range, concentration=DEFAULT_STRAND_CONCENTRATION):
    """Sweep temperature and compute RBS-exposure proxy from tube_analysis.

    Stub: not yet implemented.
    """
    require_nupack()
    raise NotImplementedError("NUPACK tube_analysis exposure sweep is not yet implemented.")


def extract_nupack_features(row, temp_range, concentration=DEFAULT_STRAND_CONCENTRATION):
    """Derive NUPACK feature vector for a single sequence."""
    sequence = row["sequence"]
    exposure_values = simulate_exposure_curve(sequence, temp_range, concentration)
    hill = fit_hill_curve(temp_range, exposure_values)
    mean_exposure = sum(exposure_values) / len(exposure_values) if exposure_values else None

    return {
        "nupack_Tm": hill["Tm"],
        "nupack_hill_coeff": hill["hill_coeff"],
        "nupack_amplitude": hill["amplitude"],
        "nupack_mean_exposure": mean_exposure,
    }


def run_nupack_pipeline(
    input_csv=DEFAULT_BALANCED_CSV,
    input_fasta=DEFAULT_BALANCED_FASTA,
    output_csv=f"{NUPACK_OUTPUT_DIR}/features.csv",
    temp_min=DEFAULT_TEMP_MIN,
    temp_max=DEFAULT_TEMP_MAX,
    temp_step=DEFAULT_TEMP_STEP,
    concentration=DEFAULT_STRAND_CONCENTRATION,
    dry_run=True,
):
    dataset = load_balanced_dataset(input_csv, input_fasta)
    temp_range = build_temp_range(temp_min, temp_max, temp_step)
    output_path = resolve_path(output_csv)

    if dry_run:
        print(f"NUPACK dry-run: {len(dataset)} sequences")
        print(f"  input CSV:   {resolve_path(input_csv)}")
        print(f"  input FASTA: {resolve_path(input_fasta)}")
        print(f"  output:      {output_path}")
        print(f"  temp range:  {temp_range[0]}–{temp_range[-1]}°C (step {temp_step})")
        print(f"  concentration: {concentration} M")
        print(f"  features:    {', '.join(NUPACK_FEATURE_COLUMNS)}")
        print("  engines:     Model + Tube + tube_analysis (NUPACK 4.1)")
        return dataset

    require_nupack()
    feature_rows = []
    for _, row in dataset.iterrows():
        features = extract_nupack_features(row, temp_range, concentration)
        feature_rows.append({**row.to_dict(), **features})

    result = write_feature_table(
        pd.DataFrame(feature_rows),
        output_path,
        NUPACK_FEATURE_COLUMNS,
    )
    print(f"Wrote NUPACK features: {result}")
    return result


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Extract NUPACK thermodynamic features via test-tube ensemble analysis."
    )
    parser.add_argument("--input-csv", default=DEFAULT_BALANCED_CSV)
    parser.add_argument("--input-fasta", default=DEFAULT_BALANCED_FASTA)
    parser.add_argument("--output-csv", default=f"{NUPACK_OUTPUT_DIR}/features.csv")
    parser.add_argument("--temp-min", type=int, default=DEFAULT_TEMP_MIN)
    parser.add_argument("--temp-max", type=int, default=DEFAULT_TEMP_MAX)
    parser.add_argument("--temp-step", type=int, default=DEFAULT_TEMP_STEP)
    parser.add_argument(
        "--concentration",
        type=float,
        default=DEFAULT_STRAND_CONCENTRATION,
        help="Single-strand concentration in M (default: 1e-8).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Validate inputs and print planned feature schema without running engines.",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Execute NUPACK simulations (requires nupack installed).",
    )
    return parser


def main():
    args = _build_parser().parse_args()
    run_nupack_pipeline(
        input_csv=args.input_csv,
        input_fasta=args.input_fasta,
        output_csv=args.output_csv,
        temp_min=args.temp_min,
        temp_max=args.temp_max,
        temp_step=args.temp_step,
        concentration=args.concentration,
        dry_run=not args.run,
    )


if __name__ == "__main__":
    main()
