import argparse
import re
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

RNA_PATTERN = re.compile(r"^[AUGC]+$")
DEFAULT_OUTPUT = "data/processed/de_novo/smoke/generated.fasta"


def validate_sequence(sequence):
    sequence = sequence.replace("T", "U").upper()
    if not sequence:
        raise ValueError("Generated sequence is empty")
    if not RNA_PATTERN.match(sequence):
        invalid = sorted(set(sequence) - set("AUGC"))
        raise ValueError(f"Invalid RNA characters: {invalid}")
    return sequence


def write_fasta(sequences, output_path):
    output_path = resolve_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        for index, sequence in enumerate(sequences):
            handle.write(f">sample_{index}\n{sequence}\n")
    return output_path


def run_gener_rna(
    num_samples=2,
    max_new_tokens=64,
    temperature=1.0,
    top_k=250,
    strategy="top_k",
    output_fasta=DEFAULT_OUTPUT,
    dry_run=False,
):
    settings = load_llm_settings()
    cache_dir = resolve_path(settings.generna_cache_dir)

    if dry_run:
        print(f"GenerRNA dry-run: {num_samples} samples")
        print(f"  cache:      {cache_dir}")
        print(f"  output:     {resolve_path(output_fasta)}")
        print(f"  strategy:   {strategy}, temperature={temperature}, top_k={top_k}")
        print(f"  device:     {get_device()}")
        return []

    paths = ensure_generna_assets(cache_dir, hf_token=settings.hf_token)
    device = get_device()
    print(f"Loading GenerRNA on {device}...")
    tokenizer = load_tokenizer(paths["tokenizer_path"])
    model, device = load_generna_model(paths["ckpt_path"], device=device)

    raw_sequences = generate_sequences(
        model,
        tokenizer,
        device,
        num_samples=num_samples,
        max_new_tokens=max_new_tokens,
        strategy=strategy,
        temperature=temperature,
        top_k=top_k,
    )
    sequences = [validate_sequence(seq) for seq in raw_sequences]
    out_path = write_fasta(sequences, output_fasta)
    print(f"Wrote {len(sequences)} sequences to {out_path}")
    for index, sequence in enumerate(sequences):
        print(f"  sample_{index}: {len(sequence)} nt")
    return sequences


def _build_parser():
    parser = argparse.ArgumentParser(description="GenerRNA micro-generation smoke test.")
    parser.add_argument("--num-samples", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=250)
    parser.add_argument("--strategy", default="top_k", choices=["sampling", "top_k"])
    parser.add_argument("--output-fasta", default=DEFAULT_OUTPUT)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main():
    args = _build_parser().parse_args()
    run_gener_rna(
        num_samples=args.num_samples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        strategy=args.strategy,
        output_fasta=args.output_fasta,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
