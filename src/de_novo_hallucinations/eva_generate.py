"""EVA CLM generation with Option B TaxID panel, 512-seq chunks, and hard quality gates.

Uses the official `eva-generate` CLI (TaxID + mRNA). The EVA software wrapper
expands TaxIDs to Greengenes lineages before calling the model.
"""

from __future__ import annotations

import argparse
import gc
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
    gate_chunk,
    normalize_rna,
)
from validation_embedding.config import load_llm_settings
from validation_embedding.storage import ArtifactStorage, get_storage

DEFAULT_OUTPUT = "data/processed/de_novo/smoke/generated.fasta"
BATCH_FASTA = "data/processed/de_novo/generated.fasta"
RNA_PATTERN = re.compile(r"^[AUGC]+$")

_STORAGE: ArtifactStorage | None = None
_EMBEDDING_SUBDIR = ""


def _flush_storage():
    global _STORAGE
    if _STORAGE is not None:
        _STORAGE.sync_up(embedding_subdir=_EMBEDDING_SUBDIR)


def _register_sigterm_handler(storage: ArtifactStorage, embedding_subdir: str):
    global _STORAGE, _EMBEDDING_SUBDIR
    _STORAGE = storage
    _EMBEDDING_SUBDIR = embedding_subdir

    def _handle_sigterm(signum, frame):
        print("SIGTERM received — flushing EVA artifacts to storage before exit...")
        storage.write_run_state(status="preempted", event="sigterm", generator="eva")
        _flush_storage()
        raise SystemExit(143)

    signal.signal(signal.SIGTERM, _handle_sigterm)


def append_fasta_record(output_path, record_id, sequence):
    output_path = resolve_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a") as handle:
        handle.write(f">{record_id}\n{sequence}\n")


def parse_fasta(path: Path) -> list[tuple[str, str]]:
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
    override = os.environ.get("EVA_GENERATE_BIN", "").strip()
    if override:
        return override
    found = shutil.which("eva-generate")
    if found:
        return found
    # Fallback: python -m entry if package installed without console script
    return "eva-generate"


def _checkpoint_dir(settings) -> Path:
    return resolve_path(
        os.environ.get("EVA_CHECKPOINT_DIR", settings.eva_checkpoint_dir)
    )


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


def _host_completed_count(manifest_rows: list[dict], host_name: str) -> int:
    return sum(1 for row in manifest_rows if row.get("host") == host_name)


def _next_global_index(completed_ids: set[str]) -> int:
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


def _quality_config(settings) -> QualityGateConfig:
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
    # Force re-download check
    remote_ok = storage.download_file("run_state.json", path)
    if not remote_ok and storage.uses_s3:
        # upload already attempted in write_run_state; list not required
        print("WARNING: S3 probe download returned False (object may be eventually consistent)")
    print(f"S3 probe ok: bucket={storage.settings.aws_s3_bucket} "
          f"prefix={storage.settings.aws_s3_prefix}")


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
):
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
        print("EVA dry-run")
        print(f"  rna_type:   {rna}")
        print(f"  smoke:      {smoke}")
        print(f"  chunk_size: {chunk_size}")
        print(f"  target:     {total_target}")
        print(f"  checkpoint: {checkpoint}")
        print(f"  output:     {resolve_path(out)}")
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
    generated_new = []

    if not checkpoint.exists():
        raise FileNotFoundError(
            f"EVA checkpoint not found at {checkpoint}. "
            "Download with: huggingface-cli download GENTEL-Lab/EVA --local-dir "
            f"{checkpoint}"
        )

    try:
        for host in hosts:
            already = _host_completed_count(manifest_rows, host.name)
            remaining = max(host.n_seqs - already, 0)
            print(
                f"=== Host {host.name} taxid={host.taxid} "
                f"remaining={remaining}/{host.n_seqs} ==="
            )
            while remaining > 0:
                this_chunk = min(chunk_size, remaining)
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
                    )

                # Soft-skip empties before hard gate (still gate the accepted set)
                accepted_seqs = []
                for raw in raw_seqs:
                    seq = normalize_rna(raw)
                    if not seq:
                        print("  skipping empty sequence from EVA")
                        continue
                    accepted_seqs.append(seq)

                if not accepted_seqs:
                    raise QualityGateError(
                        ["Invalid Biological Formatting: EVA chunk produced zero sequences"]
                    )

                # Hard fail — stop run (orchestrator stops pod)
                gate_chunk(accepted_seqs, cfg=qcfg)

                for seq in accepted_seqs:
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
                    print(f"  {record_id} ({host.name}): {len(seq)} nt")
                    start_index += 1

                remaining -= len(accepted_seqs)
                manifest_rows = storage.load_generation_manifest()
                storage.write_run_state(
                    status="generating",
                    generator="eva",
                    host=host.name,
                    taxid=host.taxid,
                    last_chunk_status="pass",
                    gate_errors=[],
                    generated=len(storage.completed_generation_ids()),
                    target=total_target,
                )
                storage.sync_up(embedding_subdir=embedding_subdir)
                gc.collect()
                print(
                    f"Checkpoint: {len(storage.completed_generation_ids())}/{total_target} "
                    f"(host {host.name} remaining ≈ {remaining})"
                )

        storage.write_run_state(
            status="generation_complete",
            generator="eva",
            last_chunk_status="pass",
            gate_errors=[],
            generated=len(storage.completed_generation_ids()),
            target=total_target,
        )
        storage.sync_up(embedding_subdir=embedding_subdir)
        print(f"Wrote {len(generated_new)} new EVA sequences to {out_path}")
        return generated_new

    except QualityGateError as exc:
        print(f"QUALITY GATE FAILED — aborting EVA run: {exc}")
        storage.write_run_state(
            status="quality_gate_failed",
            generator="eva",
            last_chunk_status="fail",
            gate_errors=exc.errors,
            generated=len(storage.completed_generation_ids()),
            target=total_target,
        )
        storage.sync_up(embedding_subdir=embedding_subdir)
        raise


def _build_parser():
    parser = argparse.ArgumentParser(
        description="EVA CLM generation (Option B TaxID panel + chunk gates)."
    )
    parser.add_argument("--smoke", action="store_true", help="Tiny multi-host smoke quotas")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-fasta", default=None)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--skip-s3-probe", action="store_true")
    return parser


def main():
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
        # Hard abort: stop pod even if orchestrator set RUNPOD_SKIP_TERMINATE.
        if not args.dry_run:
            runpod_terminate_if_configured(force=True)
        raise SystemExit(2) from None
    except Exception:
        if not args.dry_run:
            runpod_terminate_if_configured(force=True)
        raise
    else:
        # Standalone success stops the pod; orchestrator sets RUNPOD_SKIP_TERMINATE.
        if not args.dry_run:
            runpod_terminate_if_configured()


if __name__ == "__main__":
    main()
