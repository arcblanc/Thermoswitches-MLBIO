#!/usr/bin/env python3
"""EC2 thermo batch orchestrator: train RF on balanced set, predict on de novo FASTA."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

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


def run_train(dry_run: bool = False) -> None:
    """Train the RF on the balanced thermo feature set."""
    run_thermo_batch(
        limit=BALANCED_LIMIT,
        batch_size=1,
        workers=2,
        resume=True,
        input_mode="balanced",
        fused_csv=FUSED_OUTPUT,
        vienna_csv=VIENNA_TRAINING,
        nupack_csv=NUPACK_TRAINING,
        dry_run=dry_run,
    )
    if dry_run:
        return
    train_random_forest(fused_csv=FUSED_OUTPUT, model_path=DEFAULT_MODEL_PATH)


def run_predict(dry_run: bool = False) -> None:
    """Score de novo sequences with the trained RF thermoswitch model."""
    run_thermo_batch(
        limit=DENOVO_LIMIT,
        batch_size=1,
        workers=2,
        resume=True,
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


def _build_parser() -> argparse.ArgumentParser:
    """Build the EC2 thermo-batch argument parser."""
    parser = argparse.ArgumentParser(description="EC2 thermo + Random Forest pipeline.")
    parser.add_argument("command", choices=["train", "predict"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run", action="store_true")
    return parser


def main() -> None:
    """Parse CLI args and run train or predict (dry-run unless --run)."""
    args = _build_parser().parse_args()
    dry_run = args.dry_run or not args.run
    if args.command == "train":
        run_train(dry_run=dry_run)
    elif args.command == "predict":
        run_predict(dry_run=dry_run)


if __name__ == "__main__":
    main()
