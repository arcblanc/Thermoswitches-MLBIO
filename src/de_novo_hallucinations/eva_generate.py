"""EVA CLM generation with Option B TaxID panel, chunked generate, and soft quality filters.

Uses the official `eva-generate` CLI (TaxID + mRNA). Raw CLI FASTA is persisted
before soft-drop filtering so a few low-complexity sequences cannot discard an
entire multi-hour GPU chunk.
"""

from __future__ import annotations

import argparse
import gc
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data_engineering.paths import resolve_path
from de_novo_hallucinations.eva_prompts import load_panel_hosts, panel_total, rna_type
from de_novo_hallucinations.eva_quality import (
    QualityGateConfig,
    QualityGateError,
    normalize_rna,
    soft_filter_sequences,
)
from validation_embedding.config import LLMSettings, load_llm_settings
from validation_embedding.storage import ArtifactStorage, get_storage

DEFAULT_OUTPUT = "data/processed/de_novo/smoke/generated.fasta"
BATCH_FASTA = "data/processed/de_novo/generated.fasta"
RNA_PATTERN = re.compile(r"^[AUGC]+$")

_STORAGE: ArtifactStorage | None = None
_EMBEDDING_SUBDIR = ""


def _flush_storage() -> None:
    """Upload pending EVA artifacts if a storage client is registered."""
    global _STORAGE
    if _STORAGE is not None:
        _STORAGE.sync_up(embedding_subdir=_EMBEDDING_SUBDIR)


def _register_sigterm_handler(storage: ArtifactStorage, embedding_subdir: str) -> None:
    """Flush EVA artifacts to storage when the process receives SIGTERM."""
    global _STORAGE, _EMBEDDING_SUBDIR
    _STORAGE = storage
    _EMBEDDING_SUBDIR = embedding_subdir

    def _handle_sigterm(signum: int, frame: object | None) -> None:
        """Flush artifacts to storage and exit on SIGTERM."""
        print("SIGTERM received — flushing EVA artifacts to storage before exit...")
        storage.write_run_state(status="preempted", event="sigterm", generator="eva")
        _flush_storage()
        raise SystemExit(143)

    signal.signal(signal.SIGTERM, _handle_sigterm)


def append_fasta_record(output_path: Path | str, record_id: str, sequence: str) -> None:
    """Append a single FASTA record to the output path."""
    output_path = resolve_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a") as handle:
        handle.write(f">{record_id}\n{sequence}\n")


def parse_fasta(path: Path) -> list[tuple[str, str]]:
    """Parse a FASTA file into (header, sequence) pairs."""
    records = []
    header = None
    parts: list[str] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(parts)))
                header = line[1:]
                parts = []
            else:
                parts.append(line)
    if header is not None:
        records.append((header, "".join(parts)))
    return records


def _resolve_eva_binary() -> str:
    """Return the eva-generate binary from env, PATH, or a default name."""
    override = os.environ.get("EVA_GENERATE_BIN", "").strip()
    if override:
        return override
    found = shutil.which("eva-generate")
    if found:
        return found
    return "eva-generate"


def _checkpoint_dir(settings: LLMSettings) -> Path:
    """Resolve the EVA checkpoint directory from env or settings."""
    return resolve_path(
        os.environ.get("EVA_CHECKPOINT_DIR", settings.eva_checkpoint_dir)
    )


def _raw_fasta_path(out_path: Path) -> Path:
    """Return the durable raw-generated FASTA path beside the output file."""
    return out_path.parent / "raw_generated.fasta"


def _raw_chunks_dir(out_path: Path) -> Path:
    """Return the directory used to persist per-chunk raw FASTA files."""
    return out_path.parent / "raw_chunks"


def persist_raw_chunk(
    *,
    out_path: Path,
    host_name: str,
    chunk_tag: str,
    sequences: list[str],
) -> Path:
    """Write raw EVA sequences to durable FASTA before soft filtering."""
    raw_all = _raw_fasta_path(out_path)
    raw_dir = _raw_chunks_dir(out_path)
    raw_dir.mkdir(parents=True, exist_ok=True)
    chunk_path = raw_dir / f"{host_name}_{chunk_tag}.fasta"
    raw_all.parent.mkdir(parents=True, exist_ok=True)
    with chunk_path.open("w") as chunk_handle, raw_all.open("a") as all_handle:
        for index, seq in enumerate(sequences):
            header = f"eva_raw_{host_name}_{chunk_tag}_{index}"
            block = f">{header}\n{seq}\n"
            chunk_handle.write(block)
            all_handle.write(block)
    print(f"  raw persisted: {chunk_path} (+ append {raw_all.name})")
    return chunk_path


def _request_num_seqs(remaining: int, chunk_size: int) -> int:
    """Request up to chunk_size, with a small top-up buffer when nearly done."""
    if remaining <= 0:
        return 0
    buffer_frac = float(os.environ.get("EVA_SOFT_TOPUP_BUFFER", "1.05"))
    if remaining >= chunk_size:
        return chunk_size
    buffered = int(math.ceil(remaining * max(buffer_frac, 1.0)))
    return min(chunk_size, max(remaining, buffered))


def run_eva_cli_chunk(
    *,
    checkpoint: Path,
    taxid: int,
    rna: str,
    num_seqs: int,
    output_fa: Path,
    temperature: float,
    top_k: int,
    top_p: float,
    max_length: int,
    min_length: int,
    batch_size: int = 1,
) -> list[str]:
    """Invoke eva-generate for one chunk; return RNA sequences."""
    binary = _resolve_eva_binary()
    cmd = [
        binary,
        "--checkpoint",
        str(checkpoint),
        "--format",
        "clm",
        "--rna_type",
        rna,
        "--taxid",
        str(taxid),
        "--num_seqs",
        str(num_seqs),
        "--temperature",
        str(temperature),
        "--top_k",
        str(top_k),
        "--top_p",
        str(top_p),
        "--max_length",
        str(max_length),
        "--min_length",
        str(min_length),
        "--batch_size",
        str(batch_size),
        "--output",
        str(output_fa),
    ]
    print("  $", " ".join(cmd))
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"eva-generate failed (exit {result.returncode}):\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    if not output_fa.exists():
        raise RuntimeError(f"eva-generate produced no FASTA at {output_fa}")
    return [seq for _, seq in parse_fasta(output_fa)]


def _host_completed_count(
    manifest_rows: list[dict[str, object]], host_name: str
) -> int:
    """Count manifest rows already generated for a host."""
    return sum(1 for row in manifest_rows if row.get("host") == host_name)


def _next_global_index(completed_ids: set[str]) -> int:
    """Return the next eva_sample_/sample_ index from completed ids."""
    indices = []
    for record_id in completed_ids:
        if record_id.startswith("eva_sample_"):
            try:
                indices.append(int(record_id.rsplit("_", 1)[1]))
            except ValueError:
                continue
        elif record_id.startswith("sample_"):
            try:
                indices.append(int(record_id.split("_", 1)[1]))
            except ValueError:
                continue
    return max(indices, default=-1) + 1


def _quality_config(settings: LLMSettings) -> QualityGateConfig:
    """Build quality-gate length bounds from LLM settings."""
    return QualityGateConfig(
        min_len=settings.eva_min_len,
        max_len=settings.eva_max_len,
    )


def probe_s3(storage: ArtifactStorage) -> None:
    """Put/get a tiny run_state probe to confirm S3 connectivity."""
    if not storage.uses_remote:
        print("S3 probe skipped (STORAGE_TARGET=local)")
        return
    storage.write_run_state(status="s3_probe", generator="eva", probe=True)
    path = storage.run_state_path()
    if not path.exists():
        raise RuntimeError("S3 probe failed: local run_state missing after write")
    remote_ok = storage.download_file("run_state.json", path)
    if not remote_ok and storage.uses_s3:
        print(
            "WARNING: S3 probe download returned False "
            "(object may be eventually consistent)"
        )
    print(
        f"S3 probe ok: bucket={storage.settings.aws_s3_bucket} "
        f"prefix={storage.settings.aws_s3_prefix}"
    )


def run_eva_generate(
    *,
    smoke: bool = False,
    resume: bool = False,
    dry_run: bool = False,
    output_fasta: str | None = None,
    chunk_size: int | None = None,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    skip_s3_probe: bool = False,
) -> list[str]:
    """Generate EVA sequences for the TaxID panel with resume and soft-drop."""
    settings = load_llm_settings()
    storage = get_storage(settings)
    rna = rna_type()
    hosts = load_panel_hosts(smoke=smoke)
    total_target = panel_total(hosts)
    chunk_size = chunk_size or settings.eva_chunk_size
    out = output_fasta or (DEFAULT_OUTPUT if smoke else BATCH_FASTA)
    embedding_subdir = "" if out == BATCH_FASTA else "smoke"
    checkpoint = _checkpoint_dir(settings)
    qcfg = _quality_config(settings)

    if resume:
        storage.sync_down(embedding_subdir=embedding_subdir)

    if dry_run:
        print("EVA dry-run (soft-drop + raw persist)")
        print(f"  rna_type:   {rna}")
        print(f"  smoke:      {smoke}")
        print(f"  chunk_size: {chunk_size}")
        print(f"  target:     {total_target}")
        print(f"  checkpoint: {checkpoint}")
        print(f"  output:     {resolve_path(out)}")
        print(f"  raw_fasta:  {_raw_fasta_path(resolve_path(out))}")
        print(f"  storage:    {settings.storage_target}")
        print(f"  s3_prefix:  {settings.aws_s3_prefix}")
        print(f"  len_bounds: [{settings.eva_min_len}, {settings.eva_max_len}]")
        for host in hosts:
            print(f"  panel:      {host.name} taxid={host.taxid} n={host.n_seqs}")
        return []

    if not skip_s3_probe:
        probe_s3(storage)

    _register_sigterm_handler(storage, embedding_subdir)
    completed_ids = storage.completed_generation_ids()
    manifest_rows = storage.load_generation_manifest()
    start_index = _next_global_index(completed_ids)
    out_path = resolve_path(out)
    generated_new: list[str] = []
    total_soft_dropped = 0
    zero_pass_streak = 0

    if not checkpoint.exists():
        raise FileNotFoundError(
            f"EVA checkpoint not found at {checkpoint}. "
            "Download with: huggingface-cli download GENTEL-Lab/EVA --local-dir "
            f"{checkpoint}"
        )

    for host in hosts:
        already = _host_completed_count(manifest_rows, host.name)
        remaining = max(host.n_seqs - already, 0)
        print(
            f"=== Host {host.name} taxid={host.taxid} "
            f"remaining={remaining}/{host.n_seqs} ==="
        )
        while remaining > 0:
            this_chunk = _request_num_seqs(remaining, chunk_size)
            chunk_tag = f"idx{start_index}_n{this_chunk}"
            with tempfile.TemporaryDirectory(prefix="eva_chunk_") as tmp:
                chunk_fa = Path(tmp) / "chunk.fa"
                raw_seqs = run_eva_cli_chunk(
                    checkpoint=checkpoint,
                    taxid=host.taxid,
                    rna=rna,
                    num_seqs=this_chunk,
                    output_fa=chunk_fa,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    max_length=settings.eva_max_len,
                    min_length=settings.eva_min_len,
                    batch_size=int(os.environ.get("EVA_BATCH_SIZE", "1")),
                )

            if not raw_seqs:
                raise QualityGateError(
                    ["Invalid Biological Formatting: EVA chunk produced zero sequences"]
                )

            persist_raw_chunk(
                out_path=out_path,
                host_name=host.name,
                chunk_tag=chunk_tag,
                sequences=[normalize_rna(s) or s for s in raw_seqs],
            )

            filt = soft_filter_sequences(raw_seqs, cfg=qcfg)
            total_soft_dropped += filt.n_dropped
            print(
                f"  soft_filter: kept={filt.n_passed}/{len(raw_seqs)} "
                f"dropped={filt.n_dropped} reasons={filt.drop_reason_counts}"
            )
            for index, _seq, reason in filt.dropped[:10]:
                print(f"    drop seq[{index}]: {reason}")
            if filt.n_dropped > 10:
                print(f"    … {filt.n_dropped - 10} more drops")

            if filt.n_passed == 0:
                zero_pass_streak += 1
                storage.write_run_state(
                    status="generating",
                    generator="eva",
                    host=host.name,
                    taxid=host.taxid,
                    last_chunk_status="soft_drop_all",
                    gate_errors=[r for _, _, r in filt.dropped],
                    gate_stats={
                        "mode": "soft_drop",
                        "n_raw": len(raw_seqs),
                        "n_passed": 0,
                        "n_dropped": filt.n_dropped,
                        "drop_reason_counts": filt.drop_reason_counts,
                    },
                    soft_dropped_total=total_soft_dropped,
                    generated=len(storage.completed_generation_ids()),
                    target=total_target,
                )
                storage.sync_up(embedding_subdir=embedding_subdir)
                if zero_pass_streak >= 3:
                    raise QualityGateError(
                        [
                            "Soft-drop removed every sequence in 3 consecutive "
                            "chunks — likely mode collapse; aborting"
                        ]
                        + [r for _, _, r in filt.dropped[:5]]
                    )
                print("  warning: zero sequences passed — regenerating chunk")
                gc.collect()
                continue

            zero_pass_streak = 0

            # Only keep what is still needed for this host (buffer may overshoot).
            to_keep = filt.passed[:remaining]
            accepted_buffer: list[tuple[str, str]] = []
            triage_stride = int(os.environ.get("EVA_TRIAGE_STRIDE", "250"))
            accepted_dir = out_path.parent / "accepted_chunks"
            for seq in to_keep:
                record_id = f"eva_sample_{start_index}"
                append_fasta_record(out_path, record_id, seq)
                storage.append_jsonl(
                    storage.generation_manifest_path(),
                    {
                        "record_id": record_id,
                        "sequence": seq,
                        "length_nt": len(seq),
                        "status": "generated",
                        "host": host.name,
                        "taxid": host.taxid,
                        "rna_type": rna,
                        "generator": "eva",
                    },
                )
                generated_new.append(seq)
                accepted_buffer.append((record_id, seq))
                print(f"  {record_id} ({host.name}): {len(seq)} nt")
                start_index += 1

            # Emit immutable accepted slices for stream triage / S3 (every N accepted).
            # Flush when total accepted crosses a stride boundary.
            total_accepted_now = len(storage.completed_generation_ids())
            # Write a chunk file covering this CLI pass if buffer spans a stride mark,
            # or always write per-CLI pass tagged by index range for resumability.
            if accepted_buffer and triage_stride > 0:
                first_id = accepted_buffer[0][0]
                last_id = accepted_buffer[-1][0]
                try:
                    first_idx = int(first_id.rsplit("_", 1)[1])
                    last_idx = int(last_id.rsplit("_", 1)[1])
                except ValueError:
                    first_idx, last_idx = 0, len(accepted_buffer) - 1
                # Emit when the inclusive index range crosses a multiple of stride
                # or when this batch alone is large; also emit every CLI chunk as
                # accepted_{first:04d}_{last:04d}.fasta for S3 watchers.
                chunk_path = (
                    accepted_dir / f"accepted_{first_idx:04d}_{last_idx:04d}.fasta"
                )
                accepted_dir.mkdir(parents=True, exist_ok=True)
                with chunk_path.open("w") as handle:
                    for rid, seq in accepted_buffer:
                        handle.write(f">{rid}\n{seq}\n")
                print(
                    f"  accepted_chunk: {chunk_path.name} ({len(accepted_buffer)} seqs)"
                )
                storage.write_run_state(
                    status="generating",
                    generator="eva",
                    host=host.name,
                    taxid=host.taxid,
                    last_accepted_chunk=str(chunk_path),
                    last_chunk_status="soft_pass",
                    gate_errors=[],
                    gate_stats={
                        "mode": "soft_drop",
                        "n_raw": len(raw_seqs),
                        "n_passed": len(to_keep),
                        "n_dropped": filt.n_dropped,
                        "drop_reason_counts": filt.drop_reason_counts,
                        "ok": True,
                        "accepted_chunk": chunk_path.name,
                    },
                    soft_dropped_total=total_soft_dropped,
                    generated=total_accepted_now,
                    target=total_target,
                )
            else:
                storage.write_run_state(
                    status="generating",
                    generator="eva",
                    host=host.name,
                    taxid=host.taxid,
                    last_chunk_status="soft_pass",
                    gate_errors=[],
                    gate_stats={
                        "mode": "soft_drop",
                        "n_raw": len(raw_seqs),
                        "n_passed": len(to_keep),
                        "n_dropped": filt.n_dropped,
                        "drop_reason_counts": filt.drop_reason_counts,
                        "ok": True,
                    },
                    soft_dropped_total=total_soft_dropped,
                    generated=len(storage.completed_generation_ids()),
                    target=total_target,
                )

            manifest_rows = storage.load_generation_manifest()
            already = _host_completed_count(manifest_rows, host.name)
            remaining = max(host.n_seqs - already, 0)

            storage.sync_up(embedding_subdir=embedding_subdir)
            gc.collect()
            print(
                f"Checkpoint: {len(storage.completed_generation_ids())}/{total_target} "
                f"(host {host.name} remaining ≈ {remaining})"
            )

    storage.write_run_state(
        status="generation_complete",
        generator="eva",
        last_chunk_status="soft_pass",
        gate_errors=[],
        soft_dropped_total=total_soft_dropped,
        generated=len(storage.completed_generation_ids()),
        target=total_target,
    )
    storage.sync_up(embedding_subdir=embedding_subdir)
    print(
        f"Wrote {len(generated_new)} new EVA sequences to {out_path} "
        f"(soft-dropped {total_soft_dropped} across run)"
    )
    return generated_new


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for EVA generation."""
    parser = argparse.ArgumentParser(
        description=(
            "EVA CLM generation (Option B TaxID panel + raw persist + soft-drop gates)."
        )
    )
    parser.add_argument(
        "--smoke", action="store_true", help="Tiny multi-host smoke quotas"
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-fasta", default=None)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--skip-s3-probe", action="store_true")
    return parser


def main() -> None:
    """Run EVA generation from the command line."""
    from validation_embedding.runpod_lifecycle import runpod_terminate_if_configured

    args = _build_parser().parse_args()
    try:
        run_eva_generate(
            smoke=args.smoke,
            resume=args.resume,
            dry_run=args.dry_run,
            output_fasta=args.output_fasta,
            chunk_size=args.chunk_size,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            skip_s3_probe=args.skip_s3_probe,
        )
    except QualityGateError:
        if not args.dry_run:
            runpod_terminate_if_configured(force=True)
        raise SystemExit(2) from None
    except Exception:
        if not args.dry_run:
            runpod_terminate_if_configured(force=True)
        raise
    else:
        if not args.dry_run:
            runpod_terminate_if_configured()


if __name__ == "__main__":
    main()
