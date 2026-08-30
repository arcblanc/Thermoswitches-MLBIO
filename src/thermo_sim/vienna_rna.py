import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
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
    VIENNA_OUTPUT_DIR,
    build_temp_range,
    dinucleotide_shuffle,
    extract_pair_probability,
    fit_hill_curve,
    gc_content,
    get_stable_seed,
    load_balanced_dataset,
    max_loop_length,
    max_stem_length,
    mean_unpaired_from_pair_matrix,
    mean_unpaired_in_window,
    normalize_sequence,
    positional_entropy_from_bpp,
    rbs_window_indices,
    resolve_path,
    sd_window_indices,
    window_mean,
    write_feature_table,
)

VIENNA_FEATURE_COLUMNS = [
    "viennarna_Tm",
    "viennarna_hill_coeff",
    "viennarna_amplitude",
    "viennarna_hill_bottom",
    "viennarna_hill_top",
    "viennarna_mean_unpaired_prob",
    "viennarna_sd_pair_prob_10C",
    "viennarna_sd_pair_prob_80C",
    "viennarna_MFE",
    "viennarna_max_loop_length",
    "viennarna_max_stem_length",
    "viennarna_gc_content",
    "viennarna_dangles_model",
    "viennarna_fit_status",
]

VIENNA_P_OPEN_COLUMNS = [
    "viennarna_P_open_RBS_37",
    "viennarna_P_open_RBS_55",
]

VIENNA_DYNAMIC_FEATURE_COLUMNS = [
    "viennarna_mfe_zscore",
    "viennarna_delta_P_RBS",
    "viennarna_delta_delta_G",
    "viennarna_ensemble_diversity",
    "viennarna_mean_positional_entropy",
    *VIENNA_P_OPEN_COLUMNS,
]


@dataclass
class ViennaConfig:
    dangles: int = 2
    temperature_c: float = 37.0


def require_vienna_rna() -> None:
    """Raise if ViennaRNA Python bindings are not importable."""
    try:
        import RNA  # noqa: F401
    except ImportError as exc:
        raise EnvironmentError(
            "ViennaRNA Python bindings not found. "
            "Install via bioconda or: pip install viennarna"
        ) from exc


def _build_model(config: ViennaConfig) -> object:
    """Build a ViennaRNA model-details object from config."""
    import RNA

    md = RNA.md()
    md.dangles = int(config.dangles)
    md.temperature = float(config.temperature_c)
    return md


def _partition_unpaired(sequence: str, config: ViennaConfig) -> list[float]:
    """Compute per-base unpaired probabilities at the config temperature."""
    import RNA

    md = _build_model(config)
    fc = RNA.fold_compound(sequence, md)
    fc.pf()
    bpp = fc.bpp()
    return mean_unpaired_from_pair_matrix(bpp, len(sequence))


def simulate_melting_curve(
    sequence: str,
    temp_range: list[int],
    config: ViennaConfig | None = None,
) -> list[float]:
    """Return mean unpaired probability at each temperature."""
    config = config or ViennaConfig()
    values = []
    for temp in temp_range:
        temp_config = ViennaConfig(dangles=config.dangles, temperature_c=temp)
        unpaired = _partition_unpaired(sequence, temp_config)
        values.append(sum(unpaired) / len(unpaired))
    return values


def compute_unpaired_profile(
    sequence: str, config: ViennaConfig | None = None
) -> list[float]:
    """Return the unpaired-probability profile at 37 °C by default."""
    config = config or ViennaConfig(temperature_c=37.0)
    return _partition_unpaired(sequence, config)


def pair_probability_at_sd(
    sequence: str,
    temp_range: list[int],
    sd_i: int,
    sd_j: int,
    config: ViennaConfig | None = None,
) -> list[float | None]:
    """Track Shine-Dalgarno pair probability across a temperature grid."""
    import RNA

    config = config or ViennaConfig()
    values = []
    for temp in temp_range:
        temp_config = ViennaConfig(dangles=config.dangles, temperature_c=temp)
        md = _build_model(temp_config)
        fc = RNA.fold_compound(sequence, md)
        fc.pf()
        values.append(extract_pair_probability(fc.bpp(), sd_i, sd_j))
    return values


def _fold_mfe(sequence: str, config: ViennaConfig) -> tuple[float, str]:
    """Fold the MFE structure and return (energy, dot-bracket)."""
    import RNA

    md = _build_model(config)
    fc = RNA.fold_compound(sequence, md)
    structure, mfe = fc.mfe()
    return float(mfe), str(structure)


def _pf_bundle(
    sequence: str, config: ViennaConfig
) -> tuple[float, str, float | None, np.ndarray | list]:
    """Return (mfe, structure, ensemble_diversity, bpp) from one fold_compound."""
    import RNA

    md = _build_model(config)
    fc = RNA.fold_compound(sequence, md)
    structure, mfe = fc.mfe()
    # pf() returns (centroid_or_string, free_energy); diversity via mean_bp_distance
    fc.pf()
    try:
        diversity = float(fc.mean_bp_distance())
    except Exception:
        diversity = None
    bpp = fc.bpp()
    return float(mfe), str(structure), diversity, bpp


def extract_dynamic_vienna_features(
    row: pd.Series | dict,
    *,
    n_shuffles: int = 100,
    dangles: int = 2,
    temp_low: float = 37.0,
    temp_high: float = 55.0,
    rbs_width: int = 30,
    sigma_floor: float = 1e-6,
) -> dict:
    """Composition-relative / differential Vienna features (no full melting curve)."""
    sequence = normalize_sequence(row["sequence"])
    n = len(sequence)
    if n == 0:
        empty = {col: None for col in VIENNA_DYNAMIC_FEATURE_COLUMNS}
        empty["viennarna_mfe_zscore_shuffle_mode"] = None
        empty["viennarna_MFE"] = None
        return empty

    join_key = (
        f"{row.get('rfamseq_acc', '')}|{int(row.get('seq_start', 0))}|"
        f"{int(row.get('seq_end', 0))}"
    )
    cfg37 = ViennaConfig(dangles=dangles, temperature_c=temp_low)
    cfg55 = ViennaConfig(dangles=dangles, temperature_c=temp_high)

    mfe37, _struct37, diversity, bpp37 = _pf_bundle(sequence, cfg37)
    mfe55, _struct55, _div55, bpp55 = _pf_bundle(sequence, cfg55)

    unpaired37 = mean_unpaired_from_pair_matrix(bpp37, n)
    unpaired55 = mean_unpaired_from_pair_matrix(bpp55, n)
    rbs_idx = rbs_window_indices(sequence, width=rbs_width)
    p_rbs_37 = mean_unpaired_in_window(unpaired37, rbs_idx)
    p_rbs_55 = mean_unpaired_in_window(unpaired55, rbs_idx)
    delta_p = (
        None if p_rbs_37 is None or p_rbs_55 is None else float(p_rbs_55 - p_rbs_37)
    )

    mean_s = positional_entropy_from_bpp(bpp37, n)

    rng = random.Random(get_stable_seed(join_key))
    shuffle_mfes = []
    modes = []
    for _ in range(int(n_shuffles)):
        shuf, mode = dinucleotide_shuffle(sequence, rng)
        modes.append(mode)
        mfe_s, _ = _fold_mfe(shuf, cfg37)
        shuffle_mfes.append(mfe_s)
    mu = float(np.mean(shuffle_mfes)) if shuffle_mfes else 0.0
    sigma = float(np.std(shuffle_mfes, ddof=0)) if shuffle_mfes else 0.0
    sigma = max(sigma, sigma_floor)
    zscore = float((mfe37 - mu) / sigma)
    shuffle_mode = "dinuc" if modes.count("dinuc") >= modes.count("mono") else "mono"

    return {
        "viennarna_mfe_zscore": zscore,
        "viennarna_delta_P_RBS": delta_p,
        "viennarna_delta_delta_G": float(mfe55 - mfe37),
        "viennarna_ensemble_diversity": diversity,
        "viennarna_mean_positional_entropy": mean_s,
        "viennarna_P_open_RBS_37": None if p_rbs_37 is None else float(p_rbs_37),
        "viennarna_P_open_RBS_55": None if p_rbs_55 is None else float(p_rbs_55),
        "viennarna_mfe_zscore_shuffle_mode": shuffle_mode,
        "viennarna_MFE": mfe37,  # keep native MFE consistent if caller wants it
    }


def extract_p_open_rbs(
    row: pd.Series | dict,
    *,
    dangles: int = 2,
    temp_low: float = 37.0,
    temp_high: float = 55.0,
    rbs_width: int = 30,
) -> dict:
    """Cheap RBS unpaired probabilities at 37/55 °C (no dinucleotide shuffles)."""
    sequence = normalize_sequence(row["sequence"])
    n = len(sequence)
    empty = {col: None for col in VIENNA_P_OPEN_COLUMNS}
    if n == 0:
        return empty

    cfg37 = ViennaConfig(dangles=dangles, temperature_c=temp_low)
    cfg55 = ViennaConfig(dangles=dangles, temperature_c=temp_high)
    _mfe37, _struct37, _div37, bpp37 = _pf_bundle(sequence, cfg37)
    _mfe55, _struct55, _div55, bpp55 = _pf_bundle(sequence, cfg55)
    unpaired37 = mean_unpaired_from_pair_matrix(bpp37, n)
    unpaired55 = mean_unpaired_from_pair_matrix(bpp55, n)
    rbs_idx = rbs_window_indices(sequence, width=rbs_width)
    p_rbs_37 = mean_unpaired_in_window(unpaired37, rbs_idx)
    p_rbs_55 = mean_unpaired_in_window(unpaired55, rbs_idx)
    return {
        "viennarna_P_open_RBS_37": None if p_rbs_37 is None else float(p_rbs_37),
        "viennarna_P_open_RBS_55": None if p_rbs_55 is None else float(p_rbs_55),
    }


def extract_vienna_features(
    row: pd.Series,
    temp_range: list[int],
    config: ViennaConfig | None = None,
) -> dict:
    """Extract Hill, MFE, and unpaired ViennaRNA features for one sequence."""
    config = config or ViennaConfig()
    sequence = row["sequence"]

    melting_values = simulate_melting_curve(sequence, temp_range, config)
    unpaired_profile = compute_unpaired_profile(
        sequence, ViennaConfig(dangles=config.dangles)
    )
    sd_window = sd_window_indices(sequence)
    hill_input = [
        window_mean(
            _partition_unpaired(
                sequence, ViennaConfig(dangles=config.dangles, temperature_c=temp)
            ),
            sd_window,
        )
        for temp in temp_range
    ]
    hill = fit_hill_curve(temp_range, hill_input)

    temp_to_value = dict(zip(temp_range, hill_input))
    mfe_energy, dot_bracket = _fold_mfe(sequence, ViennaConfig(dangles=config.dangles))
    return {
        "viennarna_Tm": hill["Tm"],
        "viennarna_hill_coeff": hill["hill_coeff"],
        "viennarna_amplitude": hill["amplitude"],
        "viennarna_hill_bottom": hill.get("bottom"),
        "viennarna_hill_top": hill.get("top"),
        "viennarna_mean_unpaired_prob": sum(unpaired_profile) / len(unpaired_profile),
        "viennarna_sd_pair_prob_10C": temp_to_value.get(
            10, hill_input[0] if hill_input else None
        ),
        "viennarna_sd_pair_prob_80C": temp_to_value.get(
            80, hill_input[-1] if hill_input else None
        ),
        "viennarna_MFE": mfe_energy,
        "viennarna_max_loop_length": max_loop_length(dot_bracket),
        "viennarna_max_stem_length": max_stem_length(dot_bracket),
        "viennarna_gc_content": gc_content(sequence),
        "viennarna_dangles_model": config.dangles,
        "viennarna_fit_status": hill["fit_status"],
        "_melting_curve": melting_values,
        "_hill_curve": hill_input,
    }


def compare_dangles_for_sequence(
    sequence: str, temp_range: list[int]
) -> tuple[int, dict[int, dict[str, object]]]:
    """Pick the dangles model with the lower Hill-fit RMSE."""
    results: dict[int, dict[str, object]] = {}
    for dangles in (2, 3):
        sd_window = sd_window_indices(sequence)
        curve = [
            window_mean(
                _partition_unpaired(
                    sequence, ViennaConfig(dangles=dangles, temperature_c=temp)
                ),
                sd_window,
            )
            for temp in temp_range
        ]
        hill = fit_hill_curve(temp_range, curve)
        results[dangles] = {"curve": curve, "hill": hill}

    def _dangle_rmse(payload: dict[str, object]) -> float:
        hill = payload.get("hill")
        if not isinstance(hill, dict):
            return float("inf")
        rmse = hill.get("rmse")
        if rmse is None:
            return float("inf")
        return float(rmse)

    winner = min(results, key=lambda d: _dangle_rmse(results[d]))
    return winner, results


def run_vienna_worker(row_dict: dict, temp_range: list[int], dangles: int) -> dict:
    """Extract public ViennaRNA features and attach melting-curve sidecar data."""
    require_vienna_rna()
    row = pd.Series(row_dict)
    features = extract_vienna_features(row, temp_range, ViennaConfig(dangles=dangles))
    public = {key: value for key, value in features.items() if not key.startswith("_")}
    public["_curves"] = {
        "melting_curve": features["_melting_curve"],
        "hill_curve": features["_hill_curve"],
        "temps": temp_range,
    }
    return public


def run_vienna_pipeline(
    input_csv: str | Path = DEFAULT_BALANCED_CSV,
    input_fasta: str | Path = DEFAULT_BALANCED_FASTA,
    output_csv: str | Path = f"{VIENNA_OUTPUT_DIR}/features.csv",
    temp_min: int = DEFAULT_TEMP_MIN,
    temp_max: int = DEFAULT_TEMP_MAX,
    temp_step: int = DEFAULT_TEMP_STEP,
    dangles: int = 2,
    dry_run: bool = True,
) -> pd.DataFrame | Path:
    """Extract ViennaRNA features for a dataset, or print a dry-run plan."""
    dataset = load_balanced_dataset(input_csv, input_fasta)
    temp_range = build_temp_range(temp_min, temp_max, temp_step)
    output_path = resolve_path(output_csv)

    if dry_run:
        print(f"ViennaRNA dry-run: {len(dataset)} sequences")
        print(f"  input CSV:   {resolve_path(input_csv)}")
        print(f"  input FASTA: {resolve_path(input_fasta)}")
        print(f"  output:      {output_path}")
        print(f"  temp range:  {temp_range[0]}–{temp_range[-1]}°C (step {temp_step})")
        print(f"  dangles:     {dangles}")
        print(f"  features:    {', '.join(VIENNA_FEATURE_COLUMNS)}")
        return dataset

    require_vienna_rna()
    feature_rows = []
    for _, row in dataset.iterrows():
        features = extract_vienna_features(
            row, temp_range, ViennaConfig(dangles=dangles)
        )
        feature_rows.append(
            {
                **row.to_dict(),
                **{k: v for k, v in features.items() if not k.startswith("_")},
            }
        )

    result = write_feature_table(
        pd.DataFrame(feature_rows), output_path, VIENNA_FEATURE_COLUMNS
    )
    print(f"Wrote ViennaRNA features: {result}")
    return result


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for ViennaRNA feature extraction."""
    parser = argparse.ArgumentParser(
        description="Extract ViennaRNA thermodynamic features."
    )
    parser.add_argument("--input-csv", default=DEFAULT_BALANCED_CSV)
    parser.add_argument("--input-fasta", default=DEFAULT_BALANCED_FASTA)
    parser.add_argument("--output-csv", default=f"{VIENNA_OUTPUT_DIR}/features.csv")
    parser.add_argument("--temp-min", type=int, default=DEFAULT_TEMP_MIN)
    parser.add_argument("--temp-max", type=int, default=DEFAULT_TEMP_MAX)
    parser.add_argument("--temp-step", type=int, default=DEFAULT_TEMP_STEP)
    parser.add_argument("--dangles", type=int, choices=[2, 3], default=2)
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--run", action="store_true")
    return parser


def main() -> None:
    """Parse CLI arguments and run the ViennaRNA feature pipeline."""
    args = _build_parser().parse_args()
    run_vienna_pipeline(
        input_csv=args.input_csv,
        input_fasta=args.input_fasta,
        output_csv=args.output_csv,
        temp_min=args.temp_min,
        temp_max=args.temp_max,
        temp_step=args.temp_step,
        dangles=args.dangles,
        dry_run=not args.run,
    )


if __name__ == "__main__":
    main()
