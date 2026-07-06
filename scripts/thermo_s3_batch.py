#!/usr/bin/env python3
"""EC2 thermo orchestrator: train/predict, upload artifacts to S3, optional EC2 stop."""

from __future__ import annotations

import argparse
import os
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

from data_engineering.paths import resolve_path
from thermo_sim.thermo_batch import (
    DENOVO_FUSED_OUTPUT,
    FUSED_OUTPUT,
    run_thermo_batch,
)
from thermo_sim.thermo_classifier import (
    DEFAULT_MODEL_PATH,
    DEFAULT_PREDICTIONS,
    predict_thermoswitches,
    train_random_forest,
)
from thermo_sim.thermo_common import DEFAULT_DENOVO_FASTA

VIENNA_TRAINING = "data/processed/viennarna/features.csv"
NUPACK_TRAINING = "data/processed/nupack/features.csv"
VIENNA_DENOVO = "data/processed/viennarna/denovo_features.csv"
NUPACK_DENOVO = "data/processed/nupack/denovo_features.csv"
BALANCED_LIMIT = 2396
DENOVO_LIMIT = 10000

THERMO_TRAINING_PREFIX = "thermo/training"
THERMO_DENOVO_PREFIX = "thermo/denovo"


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _s3_client():
    import boto3

    return boto3.client("s3", region_name=os.environ.get("AWS_REGION", "us-east-1"))


def _bucket() -> str:
    bucket = os.environ.get("AWS_S3_BUCKET")
    if not bucket:
        raise ValueError("AWS_S3_BUCKET is required for thermo S3 upload")
    return bucket


def upload_to_s3(local_path: str | Path, s3_key: str) -> bool:
    local_path = resolve_path(local_path)
    if not local_path.exists():
        print(f"WARNING: skip upload, missing {local_path}")
        return False
    bucket = _bucket()
    client = _s3_client()
    client.upload_file(str(local_path), bucket, s3_key)
    print(f"Uploaded s3://{bucket}/{s3_key}")
    return True


def upload_training_artifacts() -> None:
    upload_to_s3(FUSED_OUTPUT, f"{THERMO_TRAINING_PREFIX}/fused_features.csv")
    upload_to_s3(DEFAULT_MODEL_PATH, f"{THERMO_TRAINING_PREFIX}/models/rf_thermoswitch.joblib")
    upload_to_s3(VIENNA_TRAINING, f"{THERMO_TRAINING_PREFIX}/viennarna/features.csv")
    upload_to_s3(NUPACK_TRAINING, f"{THERMO_TRAINING_PREFIX}/nupack/features.csv")


def upload_denovo_artifacts() -> None:
    upload_to_s3(DENOVO_FUSED_OUTPUT, f"{THERMO_DENOVO_PREFIX}/fused_features.csv")
    upload_to_s3(DEFAULT_PREDICTIONS, f"{THERMO_DENOVO_PREFIX}/predictions.csv")
    upload_to_s3(VIENNA_DENOVO, f"{THERMO_DENOVO_PREFIX}/viennarna/features.csv")
    upload_to_s3(NUPACK_DENOVO, f"{THERMO_DENOVO_PREFIX}/nupack/features.csv")


def _metadata(path: str) -> str | None:
    try:
        url = f"http://169.254.169.254/latest/meta-data/{path}"
        with urllib.request.urlopen(url, timeout=2) as response:
            return response.read().decode().strip()
    except Exception:
        return None


def _instance_id_from_metadata() -> str | None:
    return _metadata("instance-id")


def _region_from_metadata() -> str | None:
    # Prefer explicit region; fall back to AZ (e.g. eu-west-2a -> eu-west-2).
    region = _metadata("placement/region")
    if region:
        return region
    az = _metadata("placement/availability-zone")
    if az and len(az) > 1:
        return az[:-1]
    return None


def stop_ec2_if_configured() -> None:
    if not _env_bool("EC2_AUTO_SHUTDOWN", default=True):
        print("EC2_AUTO_SHUTDOWN=false — leaving instance running")
        return

    instance_id = os.environ.get("EC2_INSTANCE_ID") or _instance_id_from_metadata()
    if not instance_id:
        print("EC2_INSTANCE_ID missing and metadata unavailable — skip stop")
        return

    import boto3

    # Instance region may differ from S3 bucket region (AWS_REGION).
    region = (
        os.environ.get("EC2_REGION")
        or _region_from_metadata()
        or os.environ.get("AWS_REGION", "us-east-1")
    )
    client = boto3.client("ec2", region_name=region)
    print(f"Stopping EC2 instance {instance_id} in {region}...")
    try:
        client.stop_instances(InstanceIds=[instance_id])
        print(f"Stop requested for {instance_id}")
    except Exception as exc:
        print(f"ERROR: failed to stop instance {instance_id} in {region}: {exc}")
        raise


def run_train(
    dry_run=False,
    resume=True,
    workers=2,
    batch_size=1,
    limit=BALANCED_LIMIT,
):
    run_thermo_batch(
        limit=limit,
        batch_size=batch_size,
        workers=workers,
        resume=resume,
        input_mode="balanced",
        fused_csv=FUSED_OUTPUT,
        vienna_csv=VIENNA_TRAINING,
        nupack_csv=NUPACK_TRAINING,
        dry_run=dry_run,
    )
    if dry_run:
        return
    train_random_forest(fused_csv=FUSED_OUTPUT, model_path=DEFAULT_MODEL_PATH)
    upload_training_artifacts()
    stop_ec2_if_configured()


def run_predict(
    dry_run=False,
    resume=True,
    workers=2,
    batch_size=1,
    limit=DENOVO_LIMIT,
):
    run_thermo_batch(
        limit=limit,
        batch_size=batch_size,
        workers=workers,
        resume=resume,
        input_mode="fasta",
        input_fasta=DEFAULT_DENOVO_FASTA,
        fused_csv=DENOVO_FUSED_OUTPUT,
        vienna_csv=VIENNA_DENOVO,
        nupack_csv=NUPACK_DENOVO,
        dry_run=dry_run,
    )
    if dry_run:
        return
    predict_thermoswitches(
        fused_csv=DENOVO_FUSED_OUTPUT,
        model_path=DEFAULT_MODEL_PATH,
        predictions_csv=DEFAULT_PREDICTIONS,
    )
    upload_denovo_artifacts()
    stop_ec2_if_configured()


def _build_parser():
    parser = argparse.ArgumentParser(
        description="EC2 thermo + Random Forest pipeline with S3 upload and optional stop."
    )
    parser.add_argument("command", choices=["train", "predict"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    return parser


def main():
    args = _build_parser().parse_args()
    dry_run = args.dry_run or not args.run
    resume = not args.no_resume
    if args.command == "train":
        run_train(
            dry_run=dry_run,
            resume=resume,
            workers=args.workers,
            batch_size=args.batch_size,
            limit=args.limit if args.limit is not None else BALANCED_LIMIT,
        )
    elif args.command == "predict":
        run_predict(
            dry_run=dry_run,
            resume=resume,
            workers=args.workers,
            batch_size=args.batch_size,
            limit=args.limit if args.limit is not None else DENOVO_LIMIT,
        )


if __name__ == "__main__":
    main()
