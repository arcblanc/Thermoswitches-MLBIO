import argparse
import gc
import json
import re
import signal
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data_engineering.paths import resolve_path
from de_novo_hallucinations.genererna.assets import ensure_generna_assets
from de_novo_hallucinations.genererna.generate import generate_sequences
from de_novo_hallucinations.genererna.loader import get_device, load_generna_model, load_tokenizer
from validation_embedding.config import load_llm_settings
from validation_embedding.storage import ArtifactStorage, get_storage

RNA_PATTERN = re.compile(r"^[AUGC]+$")
DEFAULT_OUTPUT = "data/processed/de_novo/smoke/generated.fasta"
BATCH_FASTA = "data/processed/de_novo/generated.fasta"

_STORAGE: ArtifactStorage | None = None
_EMBEDDING_SUBDIR = ""


def validate_sequence(sequence):
    sequence = sequence.replace("T", "U").upper()
    if not sequence:
        raise ValueError("Generated sequence is empty")
    if not RNA_PATTERN.match(sequence):
        invalid = sorted(set(sequence) - set("AUGC"))
        raise ValueError(f"Invalid RNA characters: {invalid}")
    return sequence


def try_validate_sequence(sequence):
    try:
        return validate_sequence(sequence)
    except ValueError as exc:
        print(f"  skipping invalid sequence: {exc}")
        return None


def append_fasta_record(output_path, record_id, sequence):
    output_path = resolve_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a") as handle:
        handle.write(f">{record_id}\n{sequence}\n")


def write_fasta(sequences, output_path, start_index=0):
    output_path = resolve_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        for offset, sequence in enumerate(sequences):
            handle.write(f">sample_{start_index + offset}\n{sequence}\n")
    return output_path


def _next_sample_index(completed_ids: set[str]) -> int:
    indices = []
    for record_id in completed_ids:
        if record_id.startswith("sample_"):
            try:
                indices.append(int(record_id.split("_", 1)[1]))
            except ValueError:
                continue
    return max(indices, default=-1) + 1


def _flush_storage():
    global _STORAGE
    if _STORAGE is not None:
        _STORAGE.sync_up(embedding_subdir=_EMBEDDING_SUBDIR)


def _register_sigterm_handler(storage: ArtifactStorage, embedding_subdir: str):
    global _STORAGE, _EMBEDDING_SUBDIR
    _STORAGE = storage
    _EMBEDDING_SUBDIR = embedding_subdir

    def _handle_sigterm(signum, frame):
        print("SIGTERM received — flushing artifacts to storage before exit...")
        storage.write_run_state(status="preempted", event="sigterm")
        _flush_storage()
        raise SystemExit(143)

    signal.signal(signal.SIGTERM, _handle_sigterm)


def run_gener_rna(
    num_samples=2,
    max_new_tokens=64,
    temperature=1.0,
    top_k=250,
    strategy="top_k",
    output_fasta=DEFAULT_OUTPUT,
    batch_size=None,
    resume=False,
    dry_run=False,
):
    settings = load_llm_settings()
    storage = get_storage(settings)
    cache_dir = resolve_path(settings.generna_cache_dir)
    batch_size = batch_size or settings.generna_batch_size
    embedding_subdir = "" if output_fasta == BATCH_FASTA else "smoke"

    if resume:
        storage.sync_down(embedding_subdir=embedding_subdir)

    completed_ids = storage.completed_generation_ids() if resume else set()
    start_index = _next_sample_index(completed_ids)
    remaining = max(num_samples - len(completed_ids), 0)

    if dry_run:
        print(f"GenerRNA dry-run: target={num_samples}, remaining={remaining}")
        print(f"  cache:      {cache_dir}")
        print(f"  output:     {resolve_path(output_fasta)}")
        print(f"  strategy:   {strategy}, temperature={temperature}, top_k={top_k}")
        print(f"  batch_size: {batch_size}, resume: {resume}")
        print(f"  completed:  {len(completed_ids)}")
        print(f"  device:     {get_device()}")
        print(f"  storage:    {settings.storage_target}")
        return []

    if remaining == 0:
        print(f"Generation complete: {len(completed_ids)} sequences already in manifest")
        return []

    _register_sigterm_handler(storage, embedding_subdir)
    paths = ensure_generna_assets(cache_dir, hf_token=settings.hf_token)
    device = get_device()
    print(f"Loading GenerRNA on {device}...")
    tokenizer = load_tokenizer(paths["tokenizer_path"])
    model, device = load_generna_model(paths["ckpt_path"], device=device)

    manifest_path = storage.generation_manifest_path()
    out_path = resolve_path(output_fasta)
    generated = []

    while remaining > 0:
        chunk_size = min(batch_size, remaining)
        raw_sequences = generate_sequences(
            model,
            tokenizer,
            device,
            num_samples=chunk_size,
            max_new_tokens=max_new_tokens,
            strategy=strategy,
            temperature=temperature,
            top_k=top_k,
        )
        accepted = 0
        for raw in raw_sequences:
            sequence = try_validate_sequence(raw)
            if sequence is None:
                continue
            record_id = f"sample_{start_index}"
            append_fasta_record(out_path, record_id, sequence)
            storage.append_jsonl(
                manifest_path,
                {
                    "record_id": record_id,
                    "sequence": sequence,
                    "length_nt": len(sequence),
                    "status": "generated",
                },
            )
            generated.append(sequence)
            print(f"  {record_id}: {len(sequence)} nt")
            start_index += 1
            accepted += 1
            if accepted >= remaining:
                break

        remaining -= accepted
        storage.write_run_state(
            status="generating",
            generated=len(storage.completed_generation_ids()),
            target=num_samples,
        )
        storage.sync_up(embedding_subdir=embedding_subdir)
        gc.collect()
        print(f"Checkpoint: {len(storage.completed_generation_ids())}/{num_samples} sequences")

    print(f"Wrote {len(generated)} new sequences to {out_path}")
    storage.write_run_state(
        status="generation_complete",
        generated=len(storage.completed_generation_ids()),
        target=num_samples,
    )
    storage.sync_up(embedding_subdir=embedding_subdir)
    return generated


def _build_parser():
    parser = argparse.ArgumentParser(description="GenerRNA sequence generation.")
    parser.add_argument("--num-samples", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=250)
    parser.add_argument("--strategy", default="top_k", choices=["sampling", "top_k"])
    parser.add_argument("--output-fasta", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main():
    from validation_embedding.runpod_lifecycle import runpod_terminate_if_configured

    args = _build_parser().parse_args()
    output_fasta = args.output_fasta
    if output_fasta is None:
        output_fasta = BATCH_FASTA if args.num_samples > 2 else DEFAULT_OUTPUT
    try:
        run_gener_rna(
            num_samples=args.num_samples,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            strategy=args.strategy,
            output_fasta=output_fasta,
            batch_size=args.batch_size,
            resume=args.resume,
            dry_run=args.dry_run,
        )
    finally:
        # Standalone GenerRNA jobs stop the pod; orchestrator sets RUNPOD_SKIP_TERMINATE.
        if not args.dry_run:
            runpod_terminate_if_configured()


if __name__ == "__main__":
    main()
