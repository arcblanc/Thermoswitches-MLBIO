#!/usr/bin/env python3
"""Stream triage watcher for growing EVA FASTA (SSH rsync or S3).

Polls every --poll-seconds (default 300). When >= stride new complete sequences
appear past the watermark, runs Vienna + slice novelty + yield, then analyzes
and logs results (stream_triage_log.md, macleod_log.md, top_candidates.fasta).

Pilot 512 should be seeded: --seed-count 512 so first slice starts at index 512.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data_engineering.paths import resolve_path
from thermo_sim.extract_fasta_dynamic_features import extract_from_fasta

STREAM_ROOT = Path("data/processed/eva_stream")
DEFAULT_STATE = STREAM_ROOT / "triage_state.json"
DEFAULT_MIRROR = STREAM_ROOT / "mirror" / "generated.fasta"
DEFAULT_MASTER_DYNAMIC = STREAM_ROOT / "dynamic_features_master.csv"
DEFAULT_MASTER_BLAST = STREAM_ROOT / "blastn_hits_master.tsv"
DEFAULT_MASTER_NHMMER = STREAM_ROOT / "nhmmer_hits_master.tbl"
DEFAULT_YIELD_CSV = STREAM_ROOT / "yield_ratio_sequences.csv"
DEFAULT_YIELD_JSON = STREAM_ROOT / "yield_ratio.json"
DEFAULT_STREAM_LOG = STREAM_ROOT / "stream_triage_log.md"
DEFAULT_ANALYSIS_JSONL = STREAM_ROOT / "analysis_log.jsonl"
DEFAULT_TOP_FASTA = STREAM_ROOT / "top_candidates.fasta"
DEFAULT_MACLEOD_LOG = Path("cluster/macleod_log.md")
PILOT_YIELD = 0.0605  # locked Path A reference
RNA_OK = re.compile(r"^[AUGCT]+$", re.I)

# F5 tunnel flaps are common; keep polling rather than aborting quickly.
MAX_SOURCE_FAILURES = 48  # ~4h at default 300s poll
PYTHON = str(ROOT / ".venv" / "bin" / "python")
if not Path(PYTHON).exists():
    PYTHON = sys.executable
TOP_N_LOG = 20


def _run_yield(**kwargs: object) -> dict[str, object]:
    """Call eva_yield_ratio.compute_yield_ratio via import from scripts path."""
    scripts = ROOT / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    from eva_yield_ratio import compute_yield_ratio

    return compute_yield_ratio(**kwargs)


def parse_complete_fasta(path: Path) -> list[tuple[str, str]]:
    """Parse complete FASTA records only; drop a truncated trailing record."""
    if not path.exists():
        return []
    text = path.read_text()
    if not text.strip():
        return []

    records: list[tuple[str, str]] = []
    header: str | None = None
    parts: list[str] = []
    lines = text.splitlines()
    for line in lines:
        if line.startswith(">"):
            if header is not None:
                seq = "".join(parts).replace(" ", "").upper()
                if seq and RNA_OK.match(seq):
                    records.append((header, seq))
                # incomplete prior (empty seq) is discarded
            header = line[1:].split()[0]
            parts = []
        else:
            if header is not None:
                parts.append(line.strip())

    # Trailing record: only keep if file ends with a non-empty sequence body.
    # If the last line was a header with no body yet, drop it (rsync race).
    if header is not None:
        seq = "".join(parts).replace(" ", "").upper()
        ends_with_header = bool(lines) and lines[-1].startswith(">")
        if seq and RNA_OK.match(seq) and not ends_with_header:
            # Heuristic: if last line is incomplete RNA (very short after header
            # mid-write), still accept only if we have a full line break after
            # some sequence — empty parts already handled.
            records.append((header, seq))
        # else: truncated trailing → drop
    return records


def write_slice_fasta(path: Path, records: list[tuple[str, str]]) -> None:
    """Write complete FASTA records to path, creating parent dirs if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for rid, seq in records:
            handle.write(f">{rid}\n{seq}\n")


def load_state(path: Path, seed_count: int) -> dict[str, object]:
    """Load triage state JSON, or seed a new watermark at seed_count."""
    if path.exists():
        return json.loads(path.read_text())
    last_id = f"eva_sample_{seed_count - 1}" if seed_count > 0 else None
    return {
        "last_triaged_count": int(seed_count),
        "last_record_id": last_id,
        "chunks": [],
        "consecutive_source_failures": 0,
        "note": (
            f"Seeded at {seed_count}; pilot already triaged under eva_pilot/"
            if seed_count
            else "Seeded at 0"
        ),
    }


def save_state(path: Path, state: dict[str, object]) -> None:
    """Write triage state JSON, creating parent dirs if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2) + "\n")


def _ssh_opts(ssh_port: int, control_path: str | None) -> str:
    """Build ssh -e string. No BatchMode — cluster uses password auth."""
    parts = [
        f"ssh -p {ssh_port}",
        "-o ConnectTimeout=20",
        "-o ServerAliveInterval=30",
        "-o ServerAliveCountMax=4",
    ]
    if control_path:
        # Reuse an already-authenticated master connection (enter password once).
        parts.extend(
            [
                "-o ControlMaster=auto",
                f"-o ControlPath={control_path}",
                "-o ControlPersist=8h",
            ]
        )
    return " ".join(parts)


def rsync_fasta(
    *,
    remote: str,
    ssh_port: int,
    remote_fasta: str,
    local_fasta: Path,
    control_path: str | None = None,
) -> None:
    """Rsync the remote EVA FASTA onto the local mirror path."""
    local_fasta.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "rsync",
        "-az",
        "-e",
        _ssh_opts(ssh_port, control_path),
        f"{remote}:{remote_fasta}",
        str(local_fasta),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"rsync failed (exit {result.returncode}): {result.stderr.strip()}"
        )


def s3_sync_fasta(*, bucket: str, key: str, local_fasta: Path) -> None:
    """Copy the EVA FASTA from S3 onto the local mirror path."""
    local_fasta.parent.mkdir(parents=True, exist_ok=True)
    uri = f"s3://{bucket}/{key.lstrip('/')}"
    cmd = ["aws", "s3", "cp", uri, str(local_fasta)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"aws s3 cp failed (exit {result.returncode}): {result.stderr.strip()}"
        )


def append_master_dynamic(master: Path, slice_csv: Path) -> pd.DataFrame:
    """Append new slice rows onto the master dynamic-features CSV and return it."""
    slice_df = pd.read_csv(slice_csv)
    if master.exists():
        master_df = pd.read_csv(master)
        have = set(master_df["record_id"].astype(str))
        new = slice_df[~slice_df["record_id"].astype(str).isin(have)]
        out = pd.concat([master_df, new], ignore_index=True)
    else:
        out = slice_df
    master.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(master, index=False)
    return out


def append_text_file(master: Path, piece: Path) -> None:
    """Append a BLAST/nhmmer fragment onto a master text table."""
    master.parent.mkdir(parents=True, exist_ok=True)
    if not piece.exists():
        piece.write_text("")  # empty hits ok
    data = piece.read_text()
    if not master.exists():
        master.write_text(data)
        return
    with master.open("a") as handle:
        if data and not data.endswith("\n"):
            handle.write("\n")
        # Skip comment lines already in nhmmer headers when appending
        for line in data.splitlines():
            if line.startswith("#") and master.stat().st_size > 0:
                continue
            handle.write(line + "\n")


def run_novelty_slice(
    slice_id: str, slice_fasta: Path, out_dir: Path
) -> tuple[Path, Path]:
    """Run slice novelty search and return BLAST and nhmmer hit paths."""
    env = os.environ.copy()
    env["SLICE_ID"] = slice_id
    env["SOURCE_FASTA"] = str(slice_fasta)
    env["OUT_DIR"] = str(out_dir)
    script = ROOT / "scripts" / "run_eva_slice_novelty.sh"
    result = subprocess.run(["bash", str(script)], cwd=str(ROOT), env=env)
    if result.returncode != 0:
        raise RuntimeError(f"slice novelty failed for {slice_id}")
    blast = out_dir / f"{slice_id}_blastn_hits.tsv"
    nhmmer = out_dir / f"{slice_id}_nhmmer_hits.tbl"
    return blast, nhmmer


def _rank_passers(yield_csv: Path) -> pd.DataFrame:
    """Return yield-pass rows ranked by Z then ΔP_RBS."""
    df = pd.read_csv(yield_csv)
    if "yield_pass" not in df.columns:
        return df.iloc[0:0]
    passed = df[df["yield_pass"].astype(bool)].copy()
    if passed.empty:
        return passed
    return passed.sort_values(
        by=["viennarna_mfe_zscore", "viennarna_delta_P_RBS"],
        ascending=[True, False],
    )


def _write_top_fasta(ranked: pd.DataFrame, fasta_path: Path) -> int:
    """Write ranked passers to FASTA and return how many records were written."""
    fasta_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with fasta_path.open("w") as handle:
        for _, row in ranked.iterrows():
            seq = str(row.get("sequence", "")).strip().upper()
            rid = str(row["record_id"])
            if not seq or not RNA_OK.match(seq):
                continue
            z = row.get("viennarna_mfe_zscore")
            dp = row.get("viennarna_delta_P_RBS")
            handle.write(f">{rid} Z={z} dP_RBS={dp}\n{seq}\n")
            n += 1
    return n


def _append_text(path: Path, text: str) -> None:
    """Append text to path, adding a trailing newline if missing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(text)
        if not text.endswith("\n"):
            handle.write("\n")


def analyze_and_log_slice(
    *,
    slice_id: str,
    stream_root: Path,
    chunk_summary: dict[str, object],
    cum: dict[str, object],
    first_id: str,
    last_id: str,
    stream_log: Path | None = None,
    macleod_log: Path | None = None,
    analysis_jsonl: Path | None = None,
    top_fasta: Path | None = None,
) -> dict[str, object]:
    """Persist slice analysis JSON, ranked passers, and append human logs."""
    stream_root = resolve_path(stream_root)
    novelty_summary_path = stream_root / "novelty" / f"{slice_id}_novelty_summary.json"
    slice_yield_csv = stream_root / f"{slice_id}_yield_sequences.csv"
    slice_yield_json = stream_root / f"{slice_id}_yield.json"

    novelty = {}
    if novelty_summary_path.exists():
        novelty = json.loads(novelty_summary_path.read_text())

    ranked = (
        _rank_passers(slice_yield_csv) if slice_yield_csv.exists() else pd.DataFrame()
    )
    top_rows = []
    for _, row in ranked.head(TOP_N_LOG).iterrows():
        top_rows.append(
            {
                "record_id": str(row["record_id"]),
                "viennarna_mfe_zscore": float(row["viennarna_mfe_zscore"])
                if pd.notna(row.get("viennarna_mfe_zscore"))
                else None,
                "viennarna_delta_P_RBS": float(row["viennarna_delta_P_RBS"])
                if pd.notna(row.get("viennarna_delta_P_RBS"))
                else None,
                "E_Rfam": (
                    "inf"
                    if str(row.get("E_Rfam")) == "inf"
                    else (float(row["E_Rfam"]) if pd.notna(row.get("E_Rfam")) else None)
                ),
            }
        )

    cats = novelty.get("categories", {})
    n = int(chunk_summary.get("n") or cum.get("n_quality_gated") or 0)
    n_pass = int(chunk_summary.get("n_pass") or 0)
    y = float(chunk_summary.get("yield_ratio") or 0.0)
    rolling_n = int(cum.get("n_quality_gated") or 0)
    rolling_pass = int(cum.get("n_yield_pass") or 0)
    rolling_y = float(cum.get("rolling_yield") or cum.get("yield_ratio") or 0.0)

    analysis = {
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "slice_id": slice_id,
        "id_range": {"first": first_id, "last": last_id},
        "n": n,
        "n_pass": n_pass,
        "yield_ratio": y,
        "vs_pilot_yield": {
            "pilot": PILOT_YIELD,
            "delta_pp": round((y - PILOT_YIELD) * 100, 2),
        },
        "gates": {
            "z_le_m2": int(chunk_summary.get("z_le_m2") or 0),
            "dp_gt_0": int(chunk_summary.get("dp_gt_0") or 0),
            "novel": int(chunk_summary.get("novel") or 0),
        },
        "novelty": {
            "identical_near": cats.get("identical_near", {}),
            "remote_homolog": cats.get("remote_homolog", {}),
            "no_hit": cats.get("no_hit", {}),
            "tools": novelty.get("tools", {}),
        },
        "rolling": {
            "n": rolling_n,
            "n_pass": rolling_pass,
            "yield_ratio": rolling_y,
            "chunks": list(cum.get("chunks") or []),
        },
        "top_passers": top_rows,
        "artifacts": {
            "slice_yield_json": str(slice_yield_json),
            "slice_yield_csv": str(slice_yield_csv),
            "novelty_summary": str(novelty_summary_path),
        },
    }

    analysis_path = stream_root / f"{slice_id}_analysis.json"
    analysis_path.write_text(json.dumps(analysis, indent=2) + "\n")

    top_path = resolve_path(top_fasta or DEFAULT_TOP_FASTA)
    # Rebuild cumulative top FASTA from master yield CSV when available
    master_yield = resolve_path(DEFAULT_YIELD_CSV)
    if master_yield.exists():
        n_top = _write_top_fasta(_rank_passers(master_yield), top_path)
    else:
        n_top = _write_top_fasta(ranked, top_path)
    analysis["top_candidates_fasta"] = str(top_path)
    analysis["n_top_candidates"] = n_top
    analysis_path.write_text(json.dumps(analysis, indent=2) + "\n")

    jsonl = resolve_path(analysis_jsonl or DEFAULT_ANALYSIS_JSONL)
    _append_text(jsonl, json.dumps(analysis))

    near = cats.get("identical_near", {})
    remote = cats.get("remote_homolog", {})
    no_hit = cats.get("no_hit", {})

    def _cat_line(label: str, blob: dict[str, object]) -> str:
        """Format one novelty-category markdown table row."""
        c = blob.get("count", "?")
        f = blob.get("fraction")
        if isinstance(f, (int, float)):
            return f"| {label} | {c} | {f * 100:.1f}% |"
        return f"| {label} | {c} | — |"

    top_md = (
        "\n".join(
            f"| {r['record_id']} | {r['viennarna_mfe_zscore']:.3f} | "
            f"{r['viennarna_delta_P_RBS']:.4f} | {r['E_Rfam']} |"
            for r in top_rows[:10]
        )
        or "| — | — | — | — |"
    )

    md = f"""
### {slice_id} triage ({first_id} … {last_id}) — {analysis["timestamp_utc"]}

| Metric | Value |
|--------|-------|
| Slice yield | **{n_pass}/{n} = {y:.2%}** |
| vs pilot 6.05% | {analysis["vs_pilot_yield"]["delta_pp"]:+.2f} pp |
| Z≤−2 | {analysis["gates"]["z_le_m2"]}/{n} |
| ΔP_RBS>0 | {analysis["gates"]["dp_gt_0"]}/{n} |
| E_Rfam>1e−3 | {analysis["gates"]["novel"]}/{n} |
| Rolling yield | {rolling_pass}/{rolling_n} = {rolling_y:.2%} |

Novelty (E≤0.1):

| Category | n | % |
|----------|---|---|
{_cat_line("identical_near", near)}
{_cat_line("remote_homolog", remote)}
{_cat_line("no_hit", no_hit)}

Top passers (by Z, then ΔP_RBS):

| ID | Z | ΔP_RBS | E_Rfam |
|----|---|--------|--------|
{top_md}

Artifacts: `{analysis_path}`, `{top_path}` ({n_top} cumulative passers).
"""
    print(md)

    stream_log_path = resolve_path(stream_log or DEFAULT_STREAM_LOG)
    if not stream_log_path.exists():
        stream_log_path.write_text(
            "# EVA stream triage log\n\n"
            "Auto-appended after each Vienna + novelty + yield slice.\n"
        )
    _append_text(stream_log_path, md)

    macleod_path = resolve_path(macleod_log or DEFAULT_MACLEOD_LOG)
    if macleod_path.exists():
        marker = f"### {slice_id} triage"
        existing = macleod_path.read_text()
        if marker not in existing:
            if "## Stream triage results" not in existing:
                _append_text(macleod_path, "\n## Stream triage results\n")
            _append_text(macleod_path, md)
            print(f"Appended analysis to {macleod_path}")
        else:
            print(f"macleod_log already has {slice_id}; skipped duplicate append")
    else:
        print(f"Warning: {macleod_path} missing; stream log only", file=sys.stderr)

    print(f"Wrote analysis {analysis_path}")
    print(f"Wrote top candidates {top_path} ({n_top})")
    print(f"Appended {stream_log_path} and {jsonl}")
    return analysis


def backfill_analysis(args: argparse.Namespace) -> None:
    """Re-run analyze_and_log for every chunk already in triage_state."""
    state = load_state(resolve_path(args.state), args.seed_count)
    stream_root = resolve_path(STREAM_ROOT)
    chunks = state.get("chunks") or []
    if not chunks:
        print("No chunks in state — nothing to backfill")
        return
    cum_chunks: list[dict[str, object]] = []
    for chunk in chunks:
        cum_chunks.append(chunk)
        slice_id = chunk["chunk_id"]
        yield_json = stream_root / f"{slice_id}_yield.json"
        if not yield_json.exists():
            print(f"Skip {slice_id}: missing {yield_json}", file=sys.stderr)
            continue
        # Approximate id range from yield CSV order
        ycsv = stream_root / f"{slice_id}_yield_sequences.csv"
        first_id = last_id = "?"
        if ycsv.exists():
            ids = pd.read_csv(ycsv)["record_id"].astype(str).tolist()
            if ids:
                first_id, last_id = ids[0], ids[-1]
        rolling = {
            "n_quality_gated": sum(c.get("n", 0) for c in cum_chunks),
            "n_yield_pass": sum(c.get("n_pass", 0) for c in cum_chunks),
            "yield_ratio": (
                sum(c.get("n_pass", 0) for c in cum_chunks)
                / max(1, sum(c.get("n", 0) for c in cum_chunks))
            ),
            "rolling_yield": (
                sum(c.get("n_pass", 0) for c in cum_chunks)
                / max(1, sum(c.get("n", 0) for c in cum_chunks))
            ),
            "chunks": list(cum_chunks),
        }
        analyze_and_log_slice(
            slice_id=slice_id,
            stream_root=stream_root,
            chunk_summary=chunk,
            cum=rolling,
            first_id=first_id,
            last_id=last_id,
        )


def triage_once(args: argparse.Namespace) -> bool:
    """Return True if a slice was processed."""
    state_path = resolve_path(args.state)
    state = load_state(state_path, args.seed_count)
    mirror = resolve_path(args.mirror_fasta)
    stream_root = resolve_path(STREAM_ROOT)
    slices_dir = stream_root / "slices"
    novelty_dir = stream_root / "novelty"

    try:
        if args.source == "ssh":
            rsync_fasta(
                remote=args.remote,
                ssh_port=args.ssh_port,
                remote_fasta=args.remote_fasta,
                local_fasta=mirror,
                control_path=args.ssh_control_path or None,
            )
        else:
            key = f"{args.s3_prefix.rstrip('/')}/de_novo/generated.fasta"
            s3_sync_fasta(bucket=args.s3_bucket, key=key, local_fasta=mirror)
        state["consecutive_source_failures"] = 0
    except Exception as exc:
        state["consecutive_source_failures"] = (
            int(state.get("consecutive_source_failures", 0)) + 1
        )
        save_state(state_path, state)
        msg = str(exc)
        print(f"SOURCE ERROR ({args.source}): {exc}", file=sys.stderr)
        # Tunnel / VPN flaps: never hard-abort on connection refused.
        if "Connection refused" in msg or "Connection reset" in msg:
            print(
                "Tunnel appears down — will keep polling. "
                "Reconnect F5 / ssh ControlMaster when ready.",
                file=sys.stderr,
            )
            return False
        if state["consecutive_source_failures"] >= MAX_SOURCE_FAILURES:
            print(
                f"Aborting after {MAX_SOURCE_FAILURES} consecutive source failures.",
                file=sys.stderr,
            )
            raise SystemExit(2) from exc
        return False

    records = parse_complete_fasta(mirror)
    complete_n = len(records)
    last = int(state.get("last_triaged_count", args.seed_count))
    available = complete_n - last
    print(f"mirror complete_n={complete_n} watermark={last} pending={available}")

    flush_remainder = args.flush_remainder and available > 0
    if available < args.stride and not flush_remainder:
        save_state(state_path, state)
        return False

    take = available if flush_remainder and available < args.stride else args.stride
    slice_records = records[last : last + take]
    if not slice_records:
        save_state(state_path, state)
        return False

    chunk_num = len(state.get("chunks", [])) + 1
    slice_id = f"slice_{chunk_num:03d}"
    slice_fasta = slices_dir / f"{slice_id}.fasta"
    write_slice_fasta(slice_fasta, slice_records)
    print(
        f"=== {slice_id}: {len(slice_records)} seqs "
        f"({slice_records[0][0]} … {slice_records[-1][0]}) ==="
    )

    slice_dyn = stream_root / f"{slice_id}_dynamic_features.csv"
    extract_from_fasta(
        fasta=str(slice_fasta),
        output=str(slice_dyn),
        n_shuffles=args.n_shuffles,
        workers=args.workers,
    )
    append_master_dynamic(resolve_path(args.master_dynamic), slice_dyn)

    blast_piece, nhmmer_piece = run_novelty_slice(slice_id, slice_fasta, novelty_dir)
    append_text_file(resolve_path(args.master_blast), blast_piece)
    append_text_file(resolve_path(args.master_nhmmer), nhmmer_piece)

    # Slice-only yield for chunk stats
    chunk_summary = _run_yield(
        dynamic_csv=str(slice_dyn),
        blast_tsv=str(blast_piece),
        nhmmer_tbl=str(nhmmer_piece),
        output_csv=str(stream_root / f"{slice_id}_yield_sequences.csv"),
        output_json=str(stream_root / f"{slice_id}_yield.json"),
        chunk_id=slice_id,
        output_chunk_json=str(stream_root / f"{slice_id}_chunk_summary.json"),
    )

    # Cumulative on master
    prior = list(state.get("chunks", []))
    prior.append(chunk_summary.get("chunk", {}))
    cum = _run_yield(
        dynamic_csv=str(args.master_dynamic),
        blast_tsv=str(args.master_blast),
        nhmmer_tbl=str(args.master_nhmmer),
        output_csv=str(args.yield_csv),
        output_json=str(args.yield_json),
        prior_chunks=prior,
    )
    print(
        f"Rolling yield: {cum['n_yield_pass']}/{cum['n_quality_gated']} "
        f"= {cum['yield_ratio']:.4f}"
    )

    analyze_and_log_slice(
        slice_id=slice_id,
        stream_root=stream_root,
        chunk_summary=chunk_summary.get("chunk", chunk_summary),
        cum=cum,
        first_id=slice_records[0][0],
        last_id=slice_records[-1][0],
    )

    state["chunks"] = prior
    state["last_triaged_count"] = last + len(slice_records)
    state["last_record_id"] = slice_records[-1][0]
    save_state(state_path, state)
    return True


def _build_parser() -> argparse.ArgumentParser:
    """Build the stream-triage argument parser."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", choices=("ssh", "s3"), default="ssh")
    p.add_argument("--remote", default="t41am25@127.0.0.1")
    p.add_argument("--ssh-port", type=int, default=1024)
    p.add_argument(
        "--ssh-control-path",
        default=str(Path.home() / ".ssh" / "cm-macleod-%r@%h:%p"),
        help=(
            "SSH ControlMaster socket so password is entered once. "
            "Open a master first (see docs), then run the watcher."
        ),
    )
    p.add_argument(
        "--remote-fasta",
        default="/home/t41am25/Thermoswitches-MLBIO/data/processed/de_novo/generated.fasta",
        help="Absolute path on the cluster (do not use ~/ — local shell expands it).",
    )
    p.add_argument("--s3-bucket", default=os.environ.get("AWS_S3_BUCKET", ""))
    p.add_argument(
        "--s3-prefix",
        default=os.environ.get("AWS_S3_PREFIX", "llm-batch/eva/v1/pilot2k"),
    )
    p.add_argument(
        "--stride", type=int, default=int(os.environ.get("EVA_TRIAGE_STRIDE", "250"))
    )
    p.add_argument("--seed-count", type=int, default=512)
    p.add_argument("--poll-seconds", type=int, default=300)
    p.add_argument("--once", action="store_true")
    p.add_argument(
        "--flush-remainder",
        action="store_true",
        help="Process <stride leftover (use when generation is done).",
    )
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--n-shuffles", type=int, default=100)
    p.add_argument("--state", default=str(DEFAULT_STATE))
    p.add_argument("--mirror-fasta", default=str(DEFAULT_MIRROR))
    p.add_argument("--master-dynamic", default=str(DEFAULT_MASTER_DYNAMIC))
    p.add_argument("--master-blast", default=str(DEFAULT_MASTER_BLAST))
    p.add_argument("--master-nhmmer", default=str(DEFAULT_MASTER_NHMMER))
    p.add_argument("--yield-csv", default=str(DEFAULT_YIELD_CSV))
    p.add_argument("--yield-json", default=str(DEFAULT_YIELD_JSON))
    p.add_argument(
        "--backfill-analysis",
        action="store_true",
        help="Re-analyze + log all chunks already in triage_state (no rsync).",
    )
    return p


def main() -> None:
    """Parse CLI args and run one-shot, backfill, or poll-loop triage."""
    args = _build_parser().parse_args()
    if args.source == "s3" and not args.s3_bucket:
        raise SystemExit("--s3-bucket required for --source s3")

    if args.backfill_analysis:
        backfill_analysis(args)
        return

    # Seed state file on first run
    state_path = resolve_path(args.state)
    if not state_path.exists():
        save_state(state_path, load_state(state_path, args.seed_count))
        print(f"Seeded {state_path} at last_triaged_count={args.seed_count}")

    if args.once:
        triage_once(args)
        return

    print(
        f"Watching source={args.source} stride={args.stride} "
        f"poll={args.poll_seconds}s seed={args.seed_count}"
    )
    while True:
        triage_once(args)
        time.sleep(max(5, args.poll_seconds))


if __name__ == "__main__":
    main()
