import argparse
import os
import sys
import threading
import time
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
import psutil

SRC_ROOT = Path(__file__).resolve().parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data_engineering.cd_hit_sequence_similarity import JOIN_COLUMNS
from thermo_sim.feature_fusion import (
    fuse_engine_features,
    validate_fused_row,
    write_fused_features,
)
from thermo_sim.nupack_engine import (
    NUPACK_FEATURE_COLUMNS,
    configure_nupack_threads,
    require_nupack,
    run_nupack_worker,
)
from thermo_sim.thermo_common import (
    DEFAULT_PROTOTYPE_CSV,
    PROTOTYPE_DIR,
    PROTOTYPE_TEMP_MAX,
    PROTOTYPE_TEMP_MIN,
    PROTOTYPE_TEMP_STEP,
    build_temp_range,
    elapsed_seconds,
    export_prototype_panel,
    load_prototype_panel,
    peak_memory_mb,
    resolve_path,
    write_json,
)
from thermo_sim.vienna_rna import (
    compare_dangles_for_sequence,
    require_vienna_rna,
    run_vienna_worker,
)


def _preflight(skip_nupack: bool = False) -> dict:
    """Check Vienna/NUPACK availability and record engine versions."""
    report = {"engines": {}}
    require_vienna_rna()
    import RNA

    report["engines"]["viennarna"] = {
        "available": True,
        "version": getattr(RNA, "__version__", "unknown"),
    }

    if skip_nupack:
        report["engines"]["nupack"] = {"available": False, "skipped": True}
        return report

    require_nupack()
    import nupack

    threads = configure_nupack_threads(os.cpu_count())
    report["engines"]["nupack"] = {
        "available": True,
        "version": getattr(nupack, "__version__", "unknown"),
        "max_threads": getattr(nupack, "get_max_threads", lambda: threads)(),
        "cpu_count": os.cpu_count(),
    }
    return report


def _cpu_sampler(stop_event: threading.Event, samples: list[float]) -> None:
    """Append process-wide CPU percent samples until stop_event is set."""
    while not stop_event.is_set():
        samples.append(psutil.cpu_percent(interval=None))
        time.sleep(2)


def _run_sequence_benchmark(
    row_dict: dict,
    temp_range: list[int],
    dangles: int,
    sodium: float,
    magnesium: float,
    threads: int,
    concentration: float,
    skip_nupack: bool,
) -> dict:
    """Run Vienna and optional NUPACK timing/memory profiles for one sequence."""
    sequence_label = row_dict.get("rfam_id", "sequence")
    vienna_timed = elapsed_seconds(
        peak_memory_mb,
        run_vienna_worker,
        row_dict,
        temp_range,
        dangles,
    )
    vienna_profile = vienna_timed["result"]
    vienna_result = vienna_profile["result"]
    payload = {
        "sequence": sequence_label,
        "panel_role": row_dict.get("panel_role"),
        "seq_length": row_dict.get("seq_length", len(row_dict.get("sequence", ""))),
        "elapsed_sec": {
            "vienna": vienna_timed["elapsed_sec"],
            "total": vienna_timed["elapsed_sec"],
        },
        "vienna": {
            "features": {
                k: v for k, v in vienna_result.items() if not k.startswith("_")
            },
            "memory_mb": {
                "peak_traced_mb": vienna_profile["peak_traced_mb"],
                "rss_delta_mb": vienna_profile["rss_delta_mb"],
                "rss_after_mb": vienna_profile["rss_after_mb"],
            },
            "curves": vienna_result.get("_curves"),
        },
    }

    if not skip_nupack:
        nupack_timed = elapsed_seconds(
            peak_memory_mb,
            run_nupack_worker,
            row_dict,
            temp_range,
            sodium,
            magnesium,
            threads,
            concentration,
        )
        nupack_profile = nupack_timed["result"]
        nupack_result = nupack_profile["result"]
        payload["elapsed_sec"]["nupack"] = nupack_timed["elapsed_sec"]
        payload["elapsed_sec"]["total"] += nupack_timed["elapsed_sec"]
        payload["nupack"] = {
            "features": {
                k: v for k, v in nupack_result.items() if not k.startswith("_")
            },
            "memory_mb": {
                "peak_traced_mb": nupack_profile["peak_traced_mb"],
                "rss_delta_mb": nupack_profile["rss_delta_mb"],
                "rss_after_mb": nupack_profile["rss_after_mb"],
            },
            "curves": nupack_result.get("_curves"),
        }
    return payload


def _worker_entry(
    args: tuple[dict, list[int], int, float, float, int, float, bool],
) -> dict:
    """Unpack pool args and run a single-sequence prototype benchmark."""
    return _run_sequence_benchmark(*args)


def run_prototype_benchmark(
    workers: int | None = None,
    temp_min: int = PROTOTYPE_TEMP_MIN,
    temp_max: int = PROTOTYPE_TEMP_MAX,
    temp_step: int = PROTOTYPE_TEMP_STEP,
    vienna_dangles: str = "both",
    sodium: float = 0.05,
    magnesium: float = 0.0,
    concentration: float = 1e-8,
    skip_nupack: bool = False,
    dry_run: bool = True,
) -> pd.DataFrame | tuple[pd.DataFrame, dict]:
    """Run the 4-sequence prototype benchmark or print a dry-run plan."""
    prototype_csv = resolve_path(DEFAULT_PROTOTYPE_CSV)
    if not prototype_csv.exists():
        export_prototype_panel()
    panel = load_prototype_panel()
    temp_range = build_temp_range(temp_min, temp_max, temp_step)
    output_dir = resolve_path(PROTOTYPE_DIR)
    curves_dir = output_dir / "curves"
    curves_dir.mkdir(parents=True, exist_ok=True)

    if dry_run:
        print(f"Prototype dry-run: {len(panel)} sequences")
        for _, row in panel.iterrows():
            print(f"  - {row['panel_role']}: {row['rfam_id']} ({row['seq_length']} nt)")
        print(f"  temp range: {temp_range[0]}–{temp_range[-1]}°C (step {temp_step})")
        print(f"  output dir: {output_dir}")
        return panel

    report = _preflight(skip_nupack=skip_nupack)
    report["temp_range"] = temp_range
    report["biophysics"] = {
        "vienna_dangles_tested": vienna_dangles,
        "nupack_sodium_M": sodium,
        "nupack_magnesium_M": magnesium,
    }

    four_u_row = panel[panel["panel_role"] == "canonical_positive"].iloc[0]
    if vienna_dangles == "both":
        chosen_dangles, dangle_results = compare_dangles_for_sequence(
            four_u_row["sequence"],
            temp_range,
        )
        report["biophysics"]["chosen_vienna_dangles"] = chosen_dangles
        report["biophysics"]["dangle_comparison"] = {
            str(dangles): {
                "rmse": payload["hill"]["rmse"],
                "Tm": payload["hill"]["Tm"],
                "fit_status": payload["hill"]["fit_status"],
            }
            for dangles, payload in dangle_results.items()
        }
        for dangles, payload in dangle_results.items():
            write_json(
                curves_dir / f"vienna_fourU_d{dangles}.json",
                {
                    "temps": temp_range,
                    "curve": payload["curve"],
                    "hill": payload["hill"],
                },
            )
    else:
        chosen_dangles = int(vienna_dangles)
        report["biophysics"]["chosen_vienna_dangles"] = chosen_dangles

    workers = workers or min(4, os.cpu_count() or 1)
    threads = os.cpu_count() or 1
    row_dicts = []
    for _, row in panel.iterrows():
        row_dict = row.to_dict()
        row_dict["sequence"] = row["sequence"]
        row_dicts.append(row_dict)

    pool_args = [
        (
            row_dict,
            temp_range,
            chosen_dangles,
            sodium,
            magnesium,
            threads,
            concentration,
            skip_nupack,
        )
        for row_dict in row_dicts
    ]

    cpu_samples = []
    stop_event = threading.Event()
    sampler = threading.Thread(
        target=_cpu_sampler, args=(stop_event, cpu_samples), daemon=True
    )
    sampler.start()
    start = time.perf_counter()
    with Pool(processes=workers) as pool:
        sequence_results = pool.map(_worker_entry, pool_args)
    stop_event.set()
    sampler.join(timeout=1)
    report["runtime_sec"] = time.perf_counter() - start
    report["cpu"] = {
        "mean_cpu_pct": float(sum(cpu_samples) / len(cpu_samples))
        if cpu_samples
        else None,
        "peak_cpu_pct": float(max(cpu_samples)) if cpu_samples else None,
        "workers": workers,
    }

    vienna_rows = []
    nupack_rows = []
    report["sequences"] = []
    peak_512 = 0.0

    for row_dict, seq_result in zip(row_dicts, sequence_results):
        vienna_features = seq_result["vienna"]["features"]
        vienna_rows.append({**row_dict, **vienna_features})
        write_json(
            curves_dir / f"{row_dict['rfam_id']}_{row_dict['panel_role']}_vienna.json",
            seq_result["vienna"]["curves"],
        )

        seq_report = {
            "panel_role": row_dict["panel_role"],
            "rfam_id": row_dict["rfam_id"],
            "seq_length": row_dict["seq_length"],
            "elapsed_sec": seq_result.get("elapsed_sec"),
            "vienna": seq_result["vienna"],
        }
        if not skip_nupack:
            nupack_features = seq_result["nupack"]["features"]
            nupack_rows.append({**row_dict, **nupack_features})
            write_json(
                curves_dir
                / f"{row_dict['rfam_id']}_{row_dict['panel_role']}_nupack.json",
                seq_result["nupack"]["curves"],
            )
            seq_report["nupack"] = seq_result["nupack"]
            peak_512 = max(
                peak_512,
                seq_result["nupack"]["memory_mb"]["rss_after_mb"],
                seq_result["vienna"]["memory_mb"]["rss_after_mb"],
            )
        else:
            peak_512 = max(peak_512, seq_result["vienna"]["memory_mb"]["rss_after_mb"])
        report["sequences"].append(seq_report)

    vienna_df = pd.DataFrame(vienna_rows)
    vienna_df.to_csv(output_dir / "viennarna_features.csv", index=False)

    if skip_nupack:
        nupack_df = pd.DataFrame(
            {col: [None] * len(vienna_df) for col in NUPACK_FEATURE_COLUMNS}
        )
        for col in JOIN_COLUMNS + ["label", "panel_role", "seq_length"]:
            nupack_df[col] = vienna_df[col].values
    else:
        nupack_df = pd.DataFrame(nupack_rows)
        nupack_df.to_csv(output_dir / "nupack_features.csv", index=False)

    fused = fuse_engine_features(vienna_df, nupack_df)
    fused_path = write_fused_features(fused, output_dir / "fused_features.csv")
    report["fusion"] = {
        "rows": len(fused),
        "columns": len(fused.columns),
        "output": str(fused_path),
        "warnings": [],
    }
    for _, row in fused.iterrows():
        report["fusion"]["warnings"].extend(validate_fused_row(row))

    report["memory"] = {
        "peak_512nt_rss_mb": peak_512,
        "estimated_ram_gb_full_run": peak_512 * 2396 / 1024,
    }
    report["pass"] = {
        "fusion_rows_ok": len(fused) == 4,
        "report_written": True,
    }
    write_json(output_dir / "benchmark_report.json", report)
    print(f"Prototype complete: {fused_path}")
    print(f"Benchmark report: {output_dir / 'benchmark_report.json'}")
    return fused, report


def _build_parser() -> argparse.ArgumentParser:
    """Build the prototype benchmark CLI parser."""
    parser = argparse.ArgumentParser(
        description="Run the 4-sequence thermodynamics prototype benchmark."
    )
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--temp-min", type=int, default=PROTOTYPE_TEMP_MIN)
    parser.add_argument("--temp-max", type=int, default=PROTOTYPE_TEMP_MAX)
    parser.add_argument("--temp-step", type=int, default=PROTOTYPE_TEMP_STEP)
    parser.add_argument("--vienna-dangles", choices=["2", "3", "both"], default="both")
    parser.add_argument("--nupack-sodium", type=float, default=0.05)
    parser.add_argument("--nupack-magnesium", type=float, default=0.0)
    parser.add_argument("--concentration", type=float, default=1e-8)
    parser.add_argument("--skip-nupack", action="store_true")
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--run", action="store_true")
    return parser


def main() -> None:
    """Run the 4-sequence thermodynamics prototype from the CLI."""
    args = _build_parser().parse_args()
    run_prototype_benchmark(
        workers=args.workers,
        temp_min=args.temp_min,
        temp_max=args.temp_max,
        temp_step=args.temp_step,
        vienna_dangles=args.vienna_dangles,
        sodium=args.nupack_sodium,
        magnesium=args.nupack_magnesium,
        concentration=args.concentration,
        skip_nupack=args.skip_nupack,
        dry_run=not args.run,
    )


if __name__ == "__main__":
    main()
