#!/usr/bin/env python3
"""EVA cloud batch orchestrator: panel generate → BiRNA → verify → stop pod.

On quality-gate failure, EVA generation raises and we stop the RunPod pod.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from de_novo_hallucinations.eva_prompts import load_panel_hosts, panel_total
from validation_embedding.config import LLMSettings, load_llm_settings
from validation_embedding.runpod_lifecycle import (
    SKIP_TERMINATE_ENV,
    runpod_terminate_if_configured,
)
from validation_embedding.storage import get_storage

SMOKE_FASTA = "data/processed/de_novo/smoke/generated.fasta"
BATCH_FASTA = "data/processed/de_novo/generated.fasta"
SMOKE_EMBED_DIR = "data/processed/validation_embedding/smoke"
BATCH_EMBED_DIR = "data/processed/validation_embedding"


def _child_env() -> dict[str, str]:
    """Copy the process environment and skip child RunPod termination."""
    env = os.environ.copy()
    env[SKIP_TERMINATE_ENV] = "1"
    return env


def _remote_label(settings: LLMSettings) -> str:
    """Return a human-readable remote storage label for the run."""
    if settings.storage_target in {"s3", "runpod"}:
        return f"{settings.aws_s3_bucket}/{settings.aws_s3_prefix}"
    if settings.storage_target == "gcs":
        return f"{settings.gcs_bucket}/{settings.gcs_prefix}"
    return "local"


def run_eva_cloud_batch(*, smoke: bool = False, dry_run: bool = False) -> None:
    """Run EVA panel generation, BiRNA embedding, verify, then stop the pod."""
    settings = load_llm_settings()
    storage = get_storage(settings)
    hosts = load_panel_hosts(smoke=smoke)
    expected = panel_total(hosts)
    fasta = SMOKE_FASTA if smoke else BATCH_FASTA
    embed_dir = SMOKE_EMBED_DIR if smoke else BATCH_EMBED_DIR
    embedding_subdir = "smoke" if smoke else ""

    if dry_run:
        print("EVA cloud batch dry-run")
        print(f"  mode:         {'smoke' if smoke else 'full'}")
        print(f"  storage:      {settings.storage_target}")
        print(f"  remote:       {_remote_label(settings)}")
        print(f"  expected:     {expected}")
        print(f"  chunk_size:   {settings.eva_chunk_size}")
        print(f"  rna_type:     {settings.eva_rna_type}")
        print(f"  checkpoint:   {settings.eva_checkpoint_dir}")
        print(f"  runpod_stop:  {settings.runpod_auto_terminate}")
        for host in hosts:
            print(f"  panel:        {host.name} taxid={host.taxid} n={host.n_seqs}")
        return

    # Prefer EVA S3 prefix; warn if still on GenerRNA default.
    if settings.storage_target in {"s3", "runpod"}:
        if "eva" not in settings.aws_s3_prefix:
            print(
                f"WARNING: AWS_S3_PREFIX={settings.aws_s3_prefix!r} does not contain "
                "'eva' — consider llm-batch/eva/v1 to isolate from GenerRNA artifacts"
            )

    child_env = _child_env()
    try:
        storage.sync_down(embedding_subdir=embedding_subdir)
        storage.write_run_state(
            status="started", generator="eva", mode="smoke" if smoke else "full"
        )

        gen_cmd = [
            sys.executable,
            str(PROJECT_ROOT / "src/de_novo_hallucinations/eva_generate.py"),
            "--resume",
            "--output-fasta",
            fasta,
        ]
        if smoke:
            gen_cmd.append("--smoke")
        subprocess.run(gen_cmd, check=True, cwd=PROJECT_ROOT, env=child_env)
        storage.sync_up(embedding_subdir=embedding_subdir)

        subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "src/validation_embedding/birna_embed.py"),
                "--resume",
                "--input-fasta",
                fasta,
                "--output-subdir",
                embedding_subdir,
                "--batch-size",
                str(min(settings.generna_batch_size, settings.eva_chunk_size)),
            ],
            check=True,
            cwd=PROJECT_ROOT,
            env=child_env,
        )
        storage.sync_up(embedding_subdir=embedding_subdir)

        subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts/verify_llm_smoke_outputs.py"),
                "--expected",
                str(expected),
                "--fasta",
                fasta,
                "--embed-dir",
                embed_dir,
            ],
            check=True,
            cwd=PROJECT_ROOT,
            env=child_env,
        )

        storage.write_run_state(
            status="complete", generator="eva", mode="smoke" if smoke else "full"
        )
        storage.sync_up(embedding_subdir=embedding_subdir)
        print("EVA cloud batch complete")
    except subprocess.CalledProcessError as exc:
        storage.write_run_state(
            status="failed",
            generator="eva",
            exit_code=exc.returncode,
            mode="smoke" if smoke else "full",
        )
        storage.sync_up(embedding_subdir=embedding_subdir)
        raise
    finally:
        runpod_terminate_if_configured(force=True)


def _build_parser() -> argparse.ArgumentParser:
    """Build the EVA cloud-batch argument parser."""
    parser = argparse.ArgumentParser(description="Run EVA panel + BiRNA cloud batch.")
    parser.add_argument(
        "--smoke", action="store_true", help="Tiny smoke panel + smoke paths"
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    """Parse CLI args and run the EVA cloud batch."""
    args = _build_parser().parse_args()
    run_eva_cloud_batch(smoke=args.smoke, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
