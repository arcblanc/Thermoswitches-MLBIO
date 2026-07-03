import argparse
import gc
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
import psutil

SRC_ROOT = Path(__file__).resolve().parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data_engineering.cd_hit_sequence_similarity import JOIN_COLUMNS
from data_engineering.paths import resolve_path
from thermo_sim.feature_fusion import append_fused_features, fuse_engine_features
from thermo_sim.nupack_engine import NUPACK_FEATURE_COLUMNS, configure_nupack_threads, require_nupack, run_nupack_worker
from thermo_sim.thermo_common import (
    DEFAULT_BALANCED_CSV,
    DEFAULT_BALANCED_FASTA,
    DEFAULT_DENOVO_FASTA,
    DEFAULT_TEMP_MAX,
    DEFAULT_TEMP_MIN,
    DEFAULT_TEMP_STEP,
    FASTA_JOIN_COLUMNS,
    NUPACK_OUTPUT_DIR,
    VIENNA_OUTPUT_DIR,
    append_feature_table,
    build_temp_range,
    load_balanced_dataset,
    load_fasta_dataset,
)
from thermo_sim.vienna_rna import VIENNA_FEATURE_COLUMNS, require_vienna_rna, run_vienna_worker

BATCH_RAM_LOG = "data/processed/batch_ram_log.jsonl"
FUSED_OUTPUT = "data/processed/fused_features.csv"
DENOVO_FUSED_OUTPUT = "data/processed/denovo_fused_features.csv"


def _rss_mb():
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def _load_completed_keys(fused_csv, join_columns):
    fused_csv = resolve_path(fused_csv)
    if not fused_csv.exists() or fused_csv.stat().st_size == 0:
        return set()
    existing = pd.read_csv(fused_csv, usecols=join_columns)
    if len(join_columns) == 1:
        return set(existing[join_columns[0]].astype(str))
    return {tuple(row) for row in existing[join_columns].itertuples(index=False, name=None)}


def _row_dicts_from_dataset(dataset, limit, resume_keys, join_columns, resume=False):
    rows = []
    if resume and limit is not None:
        remaining = max(limit - len(resume_keys), 0)
        if remaining == 0:
            return rows
        process_limit = remaining
    else:
        process_limit = limit

    single_key = len(join_columns) == 1
    for _, row in dataset.iterrows():
        if single_key:
            key = str(row[join_columns[0]])
        else:
            key = tuple(row[col] for col in join_columns)
        if key in resume_keys:
            continue
        sequence = row["sequence"]
        if not _is_valid_rna_sequence(sequence):
            invalid = sorted(set(str(sequence).upper().replace("T", "U")) - set("AUGC"))
            print(f"  skipping invalid sequence (chars={invalid})")
            continue
        row_dict = row.to_dict()
        row_dict["sequence"] = sequence
        row_dict["seq_length"] = len(sequence)
        rows.append(row_dict)
        if process_limit is not None and len(rows) >= process_limit:
            break
    return rows


def _is_valid_rna_sequence(sequence):
    if not sequence:
        return False
    return set(sequence.upper().replace("T", "U")) <= set("AUGC")


def _worker_entry(args):
    row_dict, temp_range, dangles, sodium, magnesium, threads, concentration, engine = args
    sequence = row_dict.get("sequence", "")
    if not _is_valid_rna_sequence(sequence):
        invalid = sorted(set(sequence.upper().replace("T", "U")) - set("AUGC"))
        print(f"  skipping invalid sequence (chars={invalid}): len={len(sequence)}")
        return row_dict, None, None
    vienna_result = None
    nupack_result = None
    try:
        if engine in {"both", "vienna"}:
            vienna_result = run_vienna_worker(row_dict, temp_range, dangles)
        if engine in {"both", "nupack"}:
            nupack_result = run_nupack_worker(
                row_dict,
                temp_range,
                sodium,
                magnesium,
                threads,
                concentration,
            )
    except Exception as exc:
        print(f"  skipping sequence after engine error: {exc}")
        return row_dict, None, None
    return row_dict, vienna_result, nupack_result


def _process_batch(batch_rows, temp_range, dangles, sodium, magnesium, threads, concentration, engine, workers):
    pool_args = [
        (row_dict, temp_range, dangles, sodium, magnesium, threads, concentration, engine)
        for row_dict in batch_rows
    ]
    worker_count = min(workers, len(batch_rows))
    if worker_count <= 1:
        return [_worker_entry(args) for args in pool_args]
    with Pool(processes=worker_count) as pool:
        return pool.map(_worker_entry, pool_args)


def _append_batch_results(
    results, vienna_csv, nupack_csv, fused_csv, engine, join_columns, include_label
):
    vienna_rows = []
    nupack_rows = []
    for row_dict, vienna_result, nupack_result in results:
        if vienna_result is not None:
            vienna_rows.append({**row_dict, **{k: v for k, v in vienna_result.items() if not k.startswith("_")}})
        if nupack_result is not None:
            nupack_rows.append({**row_dict, **{k: v for k, v in nupack_result.items() if not k.startswith("_")}})

    extra_columns = ["seq_length"]
    if "record_id" in join_columns:
        extra_columns = ["record_id", "seq_length"]

    if vienna_rows:
        append_feature_table(
            pd.DataFrame(vienna_rows),
            vienna_csv,
            VIENNA_FEATURE_COLUMNS,
            extra_columns=extra_columns,
            join_columns=join_columns,
            include_label=include_label,
        )
    if nupack_rows:
        append_feature_table(
            pd.DataFrame(nupack_rows),
            nupack_csv,
            NUPACK_FEATURE_COLUMNS,
            extra_columns=extra_columns,
            join_columns=join_columns,
            include_label=include_label,
        )
    if engine == "both" and vienna_rows and nupack_rows:
        fused = fuse_engine_features(
            pd.DataFrame(vienna_rows),
            pd.DataFrame(nupack_rows),
            join_on=join_columns,
        )
        append_fused_features(fused, fused_csv)


def _log_ram_event(log_path, event):
    log_path = resolve_path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as handle:
        handle.write(json.dumps(event) + "\n")


def run_thermo_batch(
    limit=10,
    batch_size=2,
    workers=2,
    engine="both",
    resume=False,
    temp_min=DEFAULT_TEMP_MIN,
    temp_max=DEFAULT_TEMP_MAX,
    temp_step=DEFAULT_TEMP_STEP,
    dangles=2,
    sodium=0.05,
    magnesium=0.0,
    concentration=1e-8,
    input_mode="balanced",
    input_csv=DEFAULT_BALANCED_CSV,
    input_fasta=DEFAULT_BALANCED_FASTA,
    vienna_csv=f"{VIENNA_OUTPUT_DIR}/features.csv",
    nupack_csv=f"{NUPACK_OUTPUT_DIR}/features.csv",
    fused_csv=FUSED_OUTPUT,
    ram_log=BATCH_RAM_LOG,
    dry_run=True,
):
    if input_mode == "fasta":
        dataset = load_fasta_dataset(input_fasta)
        join_columns = FASTA_JOIN_COLUMNS
        include_label = False
    else:
        dataset = load_balanced_dataset(input_csv, input_fasta)
        join_columns = JOIN_COLUMNS
        include_label = True

    temp_range = build_temp_range(temp_min, temp_max, temp_step)
    resume_keys = _load_completed_keys(fused_csv, join_columns) if resume else set()
    row_dicts = _row_dicts_from_dataset(
        dataset, limit, resume_keys, join_columns, resume=resume
    )

    if dry_run:
        print(f"Thermo batch dry-run: {len(row_dicts)} sequences to process")
        print(f"  input_mode:  {input_mode}")
        print(f"  batch_size:  {batch_size}, workers: {workers}, engine: {engine}")
        print(f"  resume:      {resume} ({len(resume_keys)} keys already complete)")
        print(f"  temp range:  {temp_range[0]}–{temp_range[-1]}°C (step {temp_step})")
        print(f"  outputs:     {resolve_path(vienna_csv)}, {resolve_path(nupack_csv)}, {resolve_path(fused_csv)}")
        return row_dicts

    if engine in {"both", "vienna"}:
        require_vienna_rna()
    if engine in {"both", "nupack"}:
        require_nupack()
    threads = configure_nupack_threads(os.cpu_count())

    baseline_rss = _rss_mb()
    _log_ram_event(
        ram_log,
        {"event": "start", "baseline_rss_mb": baseline_rss, "sequences": len(row_dicts)},
    )

    batches = [row_dicts[i : i + batch_size] for i in range(0, len(row_dicts), batch_size)]
    for batch_idx, batch_rows in enumerate(batches, start=1):
        rss_before = _rss_mb()
        start = time.perf_counter()
        results = _process_batch(
            batch_rows,
            temp_range,
            dangles,
            sodium,
            magnesium,
            threads,
            concentration,
            engine,
            workers,
        )
        _append_batch_results(
            results,
            vienna_csv,
            nupack_csv,
            fused_csv,
            engine,
            join_columns,
            include_label,
        )
        del results
        gc.collect()
        rss_after_gc = _rss_mb()
        elapsed = time.perf_counter() - start
        event = {
            "event": "batch_complete",
            "batch_idx": batch_idx,
            "batch_size": len(batch_rows),
            "rss_before_mb": round(rss_before, 2),
            "rss_after_gc_mb": round(rss_after_gc, 2),
            "baseline_rss_mb": round(baseline_rss, 2),
            "delta_from_baseline_mb": round(rss_after_gc - baseline_rss, 2),
            "elapsed_sec": round(elapsed, 2),
        }
        _log_ram_event(ram_log, event)
        print(
            f"Batch {batch_idx}/{len(batches)}: "
            f"RSS {rss_before:.1f} -> {rss_after_gc:.1f} MB after gc "
            f"(baseline {baseline_rss:.1f} MB, +{rss_after_gc - baseline_rss:.1f} MB)"
        )

    _log_ram_event(ram_log, {"event": "complete", "batches": len(batches)})
    print(f"Batch run complete. RAM log: {resolve_path(ram_log)}")
    return row_dicts


def _build_parser():
    parser = argparse.ArgumentParser(description="Run batched thermodynamic feature extraction.")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--engine", choices=["both", "vienna", "nupack"], default="both")
    parser.add_argument("--input-mode", choices=["balanced", "fasta"], default="balanced")
    parser.add_argument("--input-csv", default=DEFAULT_BALANCED_CSV)
    parser.add_argument("--input-fasta", default=DEFAULT_BALANCED_FASTA)
    parser.add_argument("--fused-csv", default=FUSED_OUTPUT)
    parser.add_argument("--vienna-csv", default=f"{VIENNA_OUTPUT_DIR}/features.csv")
    parser.add_argument("--nupack-csv", default=f"{NUPACK_OUTPUT_DIR}/features.csv")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--temp-min", type=int, default=DEFAULT_TEMP_MIN)
    parser.add_argument("--temp-max", type=int, default=DEFAULT_TEMP_MAX)
    parser.add_argument("--temp-step", type=int, default=DEFAULT_TEMP_STEP)
    parser.add_argument("--dangles", type=int, choices=[2, 3], default=2)
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--run", action="store_true")
    return parser


def main():
    args = _build_parser().parse_args()
    input_fasta = args.input_fasta
    if args.input_mode == "fasta" and args.input_fasta == DEFAULT_BALANCED_FASTA:
        input_fasta = DEFAULT_DENOVO_FASTA
    run_thermo_batch(
        limit=args.limit,
        batch_size=args.batch_size,
        workers=args.workers,
        engine=args.engine,
        resume=args.resume,
        temp_min=args.temp_min,
        temp_max=args.temp_max,
        temp_step=args.temp_step,
        dangles=args.dangles,
        input_mode=args.input_mode,
        input_csv=args.input_csv,
        input_fasta=input_fasta,
        vienna_csv=args.vienna_csv,
        nupack_csv=args.nupack_csv,
        fused_csv=args.fused_csv,
        dry_run=not args.run,
    )


if __name__ == "__main__":
    main()
