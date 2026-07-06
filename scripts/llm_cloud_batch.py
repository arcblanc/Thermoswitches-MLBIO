#!/usr/bin/env python3
"""Cloud batch orchestrator: remote sync, generation, embedding, verify, shutdown."""

import argparse
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from validation_embedding.config import load_llm_settings
from validation_embedding.runpod_lifecycle import (
    SKIP_TERMINATE_ENV,
    runpod_terminate_if_configured,
)
from validation_embedding.storage import get_storage

BATCH_FASTA_PATH = "data/processed/de_novo/generated.fasta"
BATCH_EMBED_DIR = "data/processed/validation_embedding"


def _metadata(path: str) -> str:
    url = f"http://metadata.google.internal/computeMetadata/v1/{path}"
    request = urllib.request.Request(url, headers={"Metadata-Flavor": "Google"})
    with urllib.request.urlopen(request, timeout=5) as response:
        return response.read().decode().strip()


def vm_shutdown_if_configured():
    settings = load_llm_settings()
    if not settings.vm_auto_shutdown:
        return
    try:
        instance = _metadata("instance/name")
        zone_path = _metadata("instance/zone")
        zone = zone_path.rsplit("/", 1)[-1]
        project = _metadata("project/project-id")
    except Exception as exc:
        print(f"Not on GCE or metadata unavailable ({exc}); skipping self-delete")
        return
    print(f"Deleting instance {instance} in {zone}...")
    subprocess.run(
        [
            "gcloud",
            "compute",
            "instances",
            "delete",
            instance,
            f"--zone={zone}",
            f"--project={project}",
            "--quiet",
        ],
        check=True,
    )


def _remote_bucket_label(settings):
    if settings.storage_target == "gcs":
        return settings.gcs_bucket
    if settings.storage_target in {"s3", "runpod"}:
        return settings.aws_s3_bucket
    return None


def _child_env():
    """Children must not stop the pod; orchestrator stops once at the end."""
    env = os.environ.copy()
    env[SKIP_TERMINATE_ENV] = "1"
    return env


def run_cloud_batch(dry_run=False):
    settings = load_llm_settings()
    storage = get_storage(settings)
    embedding_subdir = ""

    if dry_run:
        print("Cloud LLM batch dry-run")
        print(f"  storage:     {settings.storage_target}")
        print(f"  bucket:      {_remote_bucket_label(settings)}")
        if settings.storage_target in {"s3", "runpod"}:
            print(f"  prefix:      {settings.aws_s3_prefix}")
        else:
            print(f"  prefix:      {settings.gcs_prefix}")
        print(f"  num_samples: {settings.generna_num_samples}")
        print(f"  batch_size:  {settings.generna_batch_size}")
        print(f"  gce_delete:  {settings.vm_auto_shutdown}")
        print(f"  runpod_stop: {settings.runpod_auto_terminate}")
        return

    # Always stop the pod when this orchestrator exits (success or failure),
    # unless RUNPOD_AUTO_TERMINATE=false.
    try:
        storage.sync_down(embedding_subdir=embedding_subdir)
        storage.write_run_state(status="started")

        child_env = _child_env()
        subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "src/de_novo_hallucinations/gener_rna.py"),
                "--resume",
                "--num-samples",
                str(settings.generna_num_samples),
                "--batch-size",
                str(settings.generna_batch_size),
                "--output-fasta",
                BATCH_FASTA_PATH,
            ],
            check=True,
            cwd=PROJECT_ROOT,
            env=child_env,
        )
        storage.sync_up(embedding_subdir=embedding_subdir)

        subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "src/validation_embedding/birna_embed.py"),
                "--resume",
                "--input-fasta",
                BATCH_FASTA_PATH,
                "--output-subdir",
                embedding_subdir,
                "--batch-size",
                str(settings.generna_batch_size),
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
                str(settings.generna_num_samples),
                "--fasta",
                BATCH_FASTA_PATH,
                "--embed-dir",
                BATCH_EMBED_DIR,
            ],
            check=True,
            cwd=PROJECT_ROOT,
            env=child_env,
        )

        storage.write_run_state(status="complete")
        storage.sync_up(embedding_subdir=embedding_subdir)
        vm_shutdown_if_configured()
    finally:
        runpod_terminate_if_configured(force=True)


def _build_parser():
    parser = argparse.ArgumentParser(description="Run GenerRNA + BiRNA cloud batch.")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main():
    args = _build_parser().parse_args()
    run_cloud_batch(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
