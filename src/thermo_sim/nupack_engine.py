import argparse
import importlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

SRC_ROOT = Path(__file__).resolve().parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from thermo_sim.thermo_common import (
    DEFAULT_BALANCED_CSV,
    DEFAULT_BALANCED_FASTA,
    DEFAULT_TEMP_MAX,
    DEFAULT_TEMP_MIN,
    DEFAULT_TEMP_STEP,
    NUPACK_OUTPUT_DIR,
    build_temp_range,
    fit_hill_curve,
    gc_content,
    load_balanced_dataset,
    local_nupack_wheel_path,
    max_loop_length,
    max_stem_length,
    normalize_sequence,
    resolve_path,
    write_feature_table,
)

NUPACK_FEATURE_COLUMNS = [
    "nupack_Tm",
    "nupack_hill_coeff",
    "nupack_amplitude",
    "nupack_hill_bottom",
    "nupack_hill_top",
    "nupack_mean_exposure",
    "nupack_MFE",
    "nupack_max_stem_length",
    "nupack_max_loop_length",
    "nupack_gc_content",
    "nupack_sd_pair_prob_10C",
    "nupack_sd_pair_prob_80C",
    "nupack_fit_status",
]

DEFAULT_STRAND_CONCENTRATION = 1e-8
DEFAULT_SODIUM = 0.05
DEFAULT_MAGNESIUM = 0.0


def import_nupack() -> Any:
    """Import the optional NUPACK Python module."""
    return importlib.import_module("nupack")


@dataclass
class NupackConfig:
    sodium: float = DEFAULT_SODIUM
    magnesium: float = DEFAULT_MAGNESIUM
    threads: int | None = None
    temperature_c: float = 37.0
    concentration: float = DEFAULT_STRAND_CONCENTRATION


def require_nupack() -> None:
    """Raise if the NUPACK Python module is not importable."""
    try:
        nupack = import_nupack()
        for name in ("Model", "SetSpec", "Strand", "Tube", "tube_analysis"):
            getattr(nupack, name)
    except ImportError as exc:
        wheel = local_nupack_wheel_path()
        hint = (
            f"pip install {wheel}" if wheel else "install from nupack-4.1.0.1/package/"
        )
        raise EnvironmentError(
            f"NUPACK Python module not found. Install the local wheel: {hint}"
        ) from exc


def configure_nupack_threads(threads: int | None = None) -> int:
    """Set NUPACK's max thread count and return the value used."""
    nupack = import_nupack()

    thread_count = threads or os.cpu_count() or 1
    if hasattr(nupack, "set_max_threads"):
        nupack.set_max_threads(thread_count)
    return thread_count


def build_model(config: NupackConfig) -> object:
    """Build a NUPACK RNA Model from salt and temperature settings."""
    Model = import_nupack().Model

    kwargs = {
        "material": "rna",
        "celsius": float(config.temperature_c),
        "sodium": float(config.sodium),
    }
    if config.magnesium:
        kwargs["magnesium"] = float(config.magnesium)
    return Model(**kwargs)


def build_single_strand_tube(
    sequence: str, concentration: float = DEFAULT_STRAND_CONCENTRATION
) -> object:
    """Build a one-strand NUPACK Tube at the given concentration."""
    nupack = import_nupack()
    SetSpec = nupack.SetSpec
    Strand = nupack.Strand
    Tube = nupack.Tube

    strand = Strand(normalize_sequence(sequence), name="utr")
    return Tube(
        strands={strand: concentration},
        complexes=SetSpec(max_size=1),
        name="single_strand",
    )


def _analyze_complex(sequence: str, config: NupackConfig) -> dict:
    """Compute MFE energy and structure metrics for a single strand."""
    nupack = import_nupack()
    Complex = nupack.Complex
    Strand = nupack.Strand
    complex_analysis = nupack.complex_analysis

    model = build_model(config)
    strand = Strand(normalize_sequence(sequence), name="utr")
    complex_obj = Complex([strand])
    result = complex_analysis(
        complexes=[complex_obj], model=model, compute=["mfe", "pairs"]
    )
    complex_result = result[complex_obj]
    structure = complex_result.mfe[0].structure
    dot_bracket = str(structure)
    return {
        "mfe": float(complex_result.mfe[0].energy),
        "max_stem_length": max_stem_length(dot_bracket),
        "max_loop_length": max_loop_length(dot_bracket),
        "tube": build_single_strand_tube(sequence, config.concentration),
        "model": model,
    }


def _exposure_at_temperature(
    sequence: str, temperature_c: float, config: NupackConfig
) -> float:
    """Return the unpaired base fraction at one temperature."""
    tube_analysis = import_nupack().tube_analysis

    temp_config = NupackConfig(
        sodium=config.sodium,
        magnesium=config.magnesium,
        threads=config.threads,
        temperature_c=temperature_c,
        concentration=config.concentration,
    )
    tube = build_single_strand_tube(sequence, temp_config.concentration)
    model = build_model(temp_config)
    analysis = tube_analysis(tubes=[tube], model=model, compute=["pairs"])
    fraction_unpaired = analysis.tubes[tube].fraction_bases_unpaired
    if fraction_unpaired is None:
        raise RuntimeError(
            "NUPACK did not return fraction_bases_unpaired; ensure compute=['pairs']."
        )
    return float(fraction_unpaired)


def simulate_exposure_curve(
    sequence: str,
    temp_range: list[int],
    config: NupackConfig | None = None,
) -> list[float]:
    """Return unpaired-base fraction at each temperature."""
    config = config or NupackConfig()
    return [_exposure_at_temperature(sequence, temp, config) for temp in temp_range]


def compute_mfe_structure(
    sequence: str,
    temperature_c: float = 37.0,
    config: NupackConfig | None = None,
) -> dict:
    """Compute MFE energy and stem/loop metrics at the given temperature."""
    config = config or NupackConfig(temperature_c=temperature_c)
    return _analyze_complex(sequence, config)


def extract_nupack_features(
    row: pd.Series,
    temp_range: list[int],
    config: NupackConfig | None = None,
) -> dict:
    """Extract Hill, exposure, and MFE NUPACK features for one sequence."""
    config = config or NupackConfig()
    sequence = row["sequence"]
    exposure_values = simulate_exposure_curve(sequence, temp_range, config)
    hill = fit_hill_curve(temp_range, exposure_values)
    mfe_info = compute_mfe_structure(sequence, temperature_c=37.0, config=config)
    temp_to_value = dict(zip(temp_range, exposure_values))

    return {
        "nupack_Tm": hill["Tm"],
        "nupack_hill_coeff": hill["hill_coeff"],
        "nupack_amplitude": hill["amplitude"],
        "nupack_hill_bottom": hill.get("bottom"),
        "nupack_hill_top": hill.get("top"),
        "nupack_mean_exposure": sum(exposure_values) / len(exposure_values),
        "nupack_MFE": mfe_info["mfe"],
        "nupack_max_stem_length": mfe_info["max_stem_length"],
        "nupack_max_loop_length": mfe_info["max_loop_length"],
        "nupack_gc_content": gc_content(sequence),
        "nupack_sd_pair_prob_10C": temp_to_value.get(
            10, exposure_values[0] if exposure_values else None
        ),
        "nupack_sd_pair_prob_80C": temp_to_value.get(
            80, exposure_values[-1] if exposure_values else None
        ),
        "nupack_fit_status": hill["fit_status"],
        "_exposure_curve": exposure_values,
    }


def run_nupack_worker(
    row_dict: dict,
    temp_range: list[int],
    sodium: float,
    magnesium: float,
    threads: int | None,
    concentration: float,
) -> dict:
    """Extract public NUPACK features and attach exposure-curve sidecar data."""
    require_nupack()
    configure_nupack_threads(threads)
    config = NupackConfig(
        sodium=sodium,
        magnesium=magnesium,
        threads=threads,
        concentration=concentration,
    )
    row = pd.Series(row_dict)
    features = extract_nupack_features(row, temp_range, config)
    public = {key: value for key, value in features.items() if not key.startswith("_")}
    public["_curves"] = {
        "exposure_curve": features["_exposure_curve"],
        "temps": temp_range,
    }
    return public


def run_nupack_pipeline(
    input_csv: str | Path = DEFAULT_BALANCED_CSV,
    input_fasta: str | Path = DEFAULT_BALANCED_FASTA,
    output_csv: str | Path = f"{NUPACK_OUTPUT_DIR}/features.csv",
    temp_min: int = DEFAULT_TEMP_MIN,
    temp_max: int = DEFAULT_TEMP_MAX,
    temp_step: int = DEFAULT_TEMP_STEP,
    concentration: float = DEFAULT_STRAND_CONCENTRATION,
    sodium: float = DEFAULT_SODIUM,
    magnesium: float = DEFAULT_MAGNESIUM,
    threads: int | None = None,
    dry_run: bool = True,
) -> pd.DataFrame | Path:
    """Extract NUPACK features for a dataset, or print a dry-run plan."""
    dataset = load_balanced_dataset(input_csv, input_fasta)
    temp_range = build_temp_range(temp_min, temp_max, temp_step)
    output_path = resolve_path(output_csv)

    if dry_run:
        wheel = local_nupack_wheel_path()
        print(f"NUPACK dry-run: {len(dataset)} sequences")
        print(f"  input CSV:   {resolve_path(input_csv)}")
        print(f"  input FASTA: {resolve_path(input_fasta)}")
        print(f"  output:      {output_path}")
        print(f"  temp range:  {temp_range[0]}–{temp_range[-1]}°C (step {temp_step})")
        print(f"  sodium:      {sodium} M")
        print(f"  magnesium:   {magnesium} M")
        print(f"  local wheel: {wheel}")
        print(f"  features:    {', '.join(NUPACK_FEATURE_COLUMNS)}")
        return dataset

    require_nupack()
    configure_nupack_threads(threads)
    config = NupackConfig(
        sodium=sodium,
        magnesium=magnesium,
        threads=threads,
        concentration=concentration,
    )
    feature_rows = []
    for _, row in dataset.iterrows():
        features = extract_nupack_features(row, temp_range, config)
        feature_rows.append(
            {
                **row.to_dict(),
                **{k: v for k, v in features.items() if not k.startswith("_")},
            }
        )

    result = write_feature_table(
        pd.DataFrame(feature_rows), output_path, NUPACK_FEATURE_COLUMNS
    )
    print(f"Wrote NUPACK features: {result}")
    return result


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for NUPACK feature extraction."""
    parser = argparse.ArgumentParser(
        description="Extract NUPACK thermodynamic features."
    )
    parser.add_argument("--input-csv", default=DEFAULT_BALANCED_CSV)
    parser.add_argument("--input-fasta", default=DEFAULT_BALANCED_FASTA)
    parser.add_argument("--output-csv", default=f"{NUPACK_OUTPUT_DIR}/features.csv")
    parser.add_argument("--temp-min", type=int, default=DEFAULT_TEMP_MIN)
    parser.add_argument("--temp-max", type=int, default=DEFAULT_TEMP_MAX)
    parser.add_argument("--temp-step", type=int, default=DEFAULT_TEMP_STEP)
    parser.add_argument(
        "--concentration", type=float, default=DEFAULT_STRAND_CONCENTRATION
    )
    parser.add_argument("--sodium", type=float, default=DEFAULT_SODIUM)
    parser.add_argument("--magnesium", type=float, default=DEFAULT_MAGNESIUM)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--run", action="store_true")
    return parser


def main() -> None:
    """Parse CLI arguments and run the NUPACK feature pipeline."""
    args = _build_parser().parse_args()
    run_nupack_pipeline(
        input_csv=args.input_csv,
        input_fasta=args.input_fasta,
        output_csv=args.output_csv,
        temp_min=args.temp_min,
        temp_max=args.temp_max,
        temp_step=args.temp_step,
        concentration=args.concentration,
        sodium=args.sodium,
        magnesium=args.magnesium,
        threads=args.threads,
        dry_run=not args.run,
    )


if __name__ == "__main__":
    main()
