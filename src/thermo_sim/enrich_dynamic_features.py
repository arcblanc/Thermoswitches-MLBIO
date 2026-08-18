"""Enrich fused features with Vienna dynamic columns (Z, ΔP_RBS, ΔΔG, Q, S).

Does not re-run NUPACK or the full melting Hill curve — only the new Vienna calls.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

SRC_ROOT = Path(__file__).resolve().parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data_engineering.cd_hit_sequence_similarity import JOIN_COLUMNS
from data_engineering.paths import resolve_path
from thermo_sim.thermo_common import load_balanced_dataset
from thermo_sim.vienna_rna import (
    VIENNA_DYNAMIC_FEATURE_COLUMNS,
    VIENNA_P_OPEN_COLUMNS,
    extract_dynamic_vienna_features,
    extract_p_open_rbs,
    require_vienna_rna,
)

DEFAULT_FUSED = "data/processed/fused_features_refseq_matched.csv"
DEFAULT_DATASET_CSV = "data/processed/balanced/length_gc_matched_refseq_dataset.csv"
DEFAULT_DATASET_FASTA = "data/processed/balanced/length_gc_matched_refseq_dataset.fasta"
DEFAULT_OUTPUT = "data/processed/fused_features_refseq_dynamic.csv"
DEFAULT_SIDECAR = "data/processed/viennarna/refseq_dynamic_features.csv"
DEFAULT_REPORT = "data/processed/refseq_dynamic_enrich_report.json"


def _join_key_tuple(row) -> tuple:
    return (str(row["rfamseq_acc"]), int(row["seq_start"]), int(row["seq_end"]))


def _worker(payload: dict) -> dict:
    feats = extract_dynamic_vienna_features(
        payload,
        n_shuffles=payload["_n_shuffles"],
        dangles=payload.get("_dangles", 2),
    )
    out = {
        "rfamseq_acc": payload["rfamseq_acc"],
        "seq_start": int(payload["seq_start"]),
        "seq_end": int(payload["seq_end"]),
        "label": int(payload["label"]) if payload.get("label") is not None else None,
    }
    for col in VIENNA_DYNAMIC_FEATURE_COLUMNS:
        out[col] = feats.get(col)
    out["viennarna_mfe_zscore_shuffle_mode"] = feats.get("viennarna_mfe_zscore_shuffle_mode")
    return out


def _p_open_worker(payload: dict) -> dict:
    feats = extract_p_open_rbs(payload, dangles=payload.get("_dangles", 2))
    out = {
        "rfamseq_acc": payload["rfamseq_acc"],
        "seq_start": int(payload["seq_start"]),
        "seq_end": int(payload["seq_end"]),
    }
    out.update(feats)
    return out


def enrich(
    fused_csv: str = DEFAULT_FUSED,
    dataset_csv: str = DEFAULT_DATASET_CSV,
    dataset_fasta: str = DEFAULT_DATASET_FASTA,
    output_csv: str = DEFAULT_OUTPUT,
    sidecar_csv: str = DEFAULT_SIDECAR,
    report_json: str = DEFAULT_REPORT,
    n_shuffles: int = 100,
    workers: int = 4,
    resume: bool = True,
    dangles: int = 2,
) -> dict:
    require_vienna_rna()
    fused = pd.read_csv(resolve_path(fused_csv))
    for c in ("seq_start", "seq_end"):
        fused[c] = fused[c].astype(int)

    dataset = load_balanced_dataset(dataset_csv, dataset_fasta)
    for c in ("seq_start", "seq_end"):
        dataset[c] = dataset[c].astype(int)

    done_keys = set()
    sidecar_path = resolve_path(sidecar_csv)
    if resume and sidecar_path.exists() and sidecar_path.stat().st_size > 0:
        prev = pd.read_csv(sidecar_path)
        for c in ("seq_start", "seq_end"):
            prev[c] = prev[c].astype(int)
        done_keys = {_join_key_tuple(r) for _, r in prev.iterrows()}
        print(f"Resume: {len(done_keys)} dynamic rows already in sidecar")
    else:
        prev = pd.DataFrame()

    todo = []
    for _, row in dataset.iterrows():
        key = _join_key_tuple(row)
        if key in done_keys:
            continue
        payload = row.to_dict()
        payload["_n_shuffles"] = n_shuffles
        payload["_dangles"] = dangles
        todo.append(payload)

    print(f"Computing dynamic features for {len(todo)} / {len(dataset)} sequences "
          f"(workers={workers}, n_shuffles={n_shuffles})")

    new_rows = []
    if todo:
        if workers <= 1:
            for i, payload in enumerate(todo, 1):
                new_rows.append(_worker(payload))
                if i % 25 == 0 or i == len(todo):
                    print(f"  {i}/{len(todo)}")
        else:
            from thermo_sim.enrich_dynamic_features import _worker as _pool_worker

            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(_pool_worker, p) for p in todo]
                for i, fut in enumerate(as_completed(futures), 1):
                    new_rows.append(fut.result())
                    if i % 25 == 0 or i == len(todo):
                        print(f"  {i}/{len(todo)}")

    if len(prev) and new_rows:
        sidecar = pd.concat([prev, pd.DataFrame(new_rows)], ignore_index=True)
    elif new_rows:
        sidecar = pd.DataFrame(new_rows)
    else:
        sidecar = prev.copy()

    sidecar = sidecar.drop_duplicates(JOIN_COLUMNS, keep="last")
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar.to_csv(sidecar_path, index=False)

    dyn_cols = VIENNA_DYNAMIC_FEATURE_COLUMNS + ["viennarna_mfe_zscore_shuffle_mode"]
    # Drop any prior dynamic cols before merge
    fused_clean = fused.drop(columns=[c for c in dyn_cols if c in fused.columns], errors="ignore")
    out = fused_clean.merge(
        sidecar[JOIN_COLUMNS + dyn_cols],
        on=JOIN_COLUMNS,
        how="left",
    )
    out_path = resolve_path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    # Sanity: Z-score means by label
    report = {
        "n_fused": int(len(out)),
        "n_sidecar": int(len(sidecar)),
        "n_computed_this_run": int(len(new_rows)),
        "n_shuffles": n_shuffles,
        "output_csv": str(out_path),
        "sidecar_csv": str(sidecar_path),
    }
    if "label" in out.columns and "viennarna_mfe_zscore" in out.columns:
        y = out["label"].astype(int)
        z = out["viennarna_mfe_zscore"]
        report["mean_z_pos"] = float(z[y == 1].mean())
        report["mean_z_neg"] = float(z[y == 0].mean())
        report["delta_mean_z_pos_minus_neg"] = report["mean_z_pos"] - report["mean_z_neg"]
        report["zscore_sanity_positives_more_negative"] = bool(
            report["mean_z_pos"] < report["mean_z_neg"]
        )
        report["n_missing_dynamic"] = int(out[VIENNA_DYNAMIC_FEATURE_COLUMNS].isna().any(axis=1).sum())
        for col in VIENNA_DYNAMIC_FEATURE_COLUMNS:
            report[f"mean_{col}"] = float(out[col].mean())

    resolve_path(report_json).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return report


def backfill_p_open(
    fused_csv: str = DEFAULT_OUTPUT,
    dataset_csv: str = DEFAULT_DATASET_CSV,
    dataset_fasta: str = DEFAULT_DATASET_FASTA,
    sidecar_csv: str = DEFAULT_SIDECAR,
    output_csv: str | None = None,
    workers: int = 4,
    dangles: int = 2,
) -> dict:
    """Fill viennarna_P_open_RBS_37/55 without re-running 100 dinucleotide shuffles."""
    require_vienna_rna()
    output_csv = output_csv or fused_csv
    fused = pd.read_csv(resolve_path(fused_csv))
    for c in ("seq_start", "seq_end"):
        fused[c] = fused[c].astype(int)

    dataset = load_balanced_dataset(dataset_csv, dataset_fasta)
    for c in ("seq_start", "seq_end"):
        dataset[c] = dataset[c].astype(int)

    sidecar_path = resolve_path(sidecar_csv)
    if sidecar_path.exists() and sidecar_path.stat().st_size > 0:
        sidecar = pd.read_csv(sidecar_path)
        for c in ("seq_start", "seq_end"):
            sidecar[c] = sidecar[c].astype(int)
    else:
        sidecar = pd.DataFrame(columns=JOIN_COLUMNS + list(VIENNA_DYNAMIC_FEATURE_COLUMNS))

    for col in VIENNA_P_OPEN_COLUMNS:
        if col not in sidecar.columns:
            sidecar[col] = pd.NA

    have = sidecar.set_index(JOIN_COLUMNS)
    need_keys = set()
    for _, row in dataset.iterrows():
        key = _join_key_tuple(row)
        if key not in have.index:
            need_keys.add(key)
            continue
        rec = have.loc[key]
        if isinstance(rec, pd.DataFrame):
            rec = rec.iloc[-1]
        if any(pd.isna(rec.get(col)) for col in VIENNA_P_OPEN_COLUMNS):
            need_keys.add(key)

    todo = []
    for _, row in dataset.iterrows():
        if _join_key_tuple(row) not in need_keys:
            continue
        payload = row.to_dict()
        payload["_dangles"] = dangles
        todo.append(payload)

    print(
        f"Backfilling P_open_RBS for {len(todo)} / {len(dataset)} sequences "
        f"(workers={workers})"
    )
    new_rows = []
    if todo:
        if workers <= 1:
            for i, payload in enumerate(todo, 1):
                new_rows.append(_p_open_worker(payload))
                if i % 25 == 0 or i == len(todo):
                    print(f"  {i}/{len(todo)}")
        else:
            from thermo_sim.enrich_dynamic_features import _p_open_worker as _pool_p_open

            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(_pool_p_open, p) for p in todo]
                for i, fut in enumerate(as_completed(futures), 1):
                    new_rows.append(fut.result())
                    if i % 25 == 0 or i == len(todo):
                        print(f"  {i}/{len(todo)}")

    if new_rows:
        patch = pd.DataFrame(new_rows)
        sidecar = sidecar.merge(patch, on=JOIN_COLUMNS, how="outer", suffixes=("", "_new"))
        for col in VIENNA_P_OPEN_COLUMNS:
            new_col = f"{col}_new"
            if new_col in sidecar.columns:
                sidecar[col] = sidecar[col].where(sidecar[col].notna(), sidecar[new_col])
                sidecar = sidecar.drop(columns=[new_col])
        sidecar = sidecar.drop_duplicates(JOIN_COLUMNS, keep="last")

    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar.to_csv(sidecar_path, index=False)

    dyn_cols = [
        c
        for c in list(VIENNA_DYNAMIC_FEATURE_COLUMNS) + ["viennarna_mfe_zscore_shuffle_mode"]
        if c in sidecar.columns
    ]
    fused_clean = fused.drop(columns=[c for c in dyn_cols if c in fused.columns], errors="ignore")
    out = fused_clean.merge(sidecar[JOIN_COLUMNS + dyn_cols], on=JOIN_COLUMNS, how="left")
    out_path = resolve_path(output_csv)
    out.to_csv(out_path, index=False)
    report = {
        "n_fused": int(len(out)),
        "n_sidecar": int(len(sidecar)),
        "n_p_open_computed": int(len(new_rows)),
        "n_missing_p_open": int(out[VIENNA_P_OPEN_COLUMNS].isna().any(axis=1).sum())
        if all(c in out.columns for c in VIENNA_P_OPEN_COLUMNS)
        else int(len(out)),
        "output_csv": str(out_path),
        "sidecar_csv": str(sidecar_path),
    }
    print(json.dumps(report, indent=2))
    return report


def main():
    p = argparse.ArgumentParser(description="Enrich fused CSV with Vienna dynamic features.")
    p.add_argument("--fused-csv", default=DEFAULT_FUSED)
    p.add_argument("--dataset-csv", default=DEFAULT_DATASET_CSV)
    p.add_argument("--dataset-fasta", default=DEFAULT_DATASET_FASTA)
    p.add_argument("--output-csv", default=DEFAULT_OUTPUT)
    p.add_argument("--sidecar-csv", default=DEFAULT_SIDECAR)
    p.add_argument("--report-json", default=DEFAULT_REPORT)
    p.add_argument("--n-shuffles", type=int, default=100)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--dangles", type=int, default=2)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument(
        "--p-open-only",
        action="store_true",
        help="Backfill P_open_RBS_37/55 without 100-shuffle Z-score recompute.",
    )
    args = p.parse_args()
    if args.p_open_only:
        backfill_p_open(
            fused_csv=args.output_csv if args.fused_csv == DEFAULT_FUSED else args.fused_csv,
            dataset_csv=args.dataset_csv,
            dataset_fasta=args.dataset_fasta,
            sidecar_csv=args.sidecar_csv,
            output_csv=args.output_csv,
            workers=args.workers,
            dangles=args.dangles,
        )
        return
    enrich(
        fused_csv=args.fused_csv,
        dataset_csv=args.dataset_csv,
        dataset_fasta=args.dataset_fasta,
        output_csv=args.output_csv,
        sidecar_csv=args.sidecar_csv,
        report_json=args.report_json,
        n_shuffles=args.n_shuffles,
        workers=args.workers,
        resume=not args.no_resume,
        dangles=args.dangles,
    )


if __name__ == "__main__":
    main()
