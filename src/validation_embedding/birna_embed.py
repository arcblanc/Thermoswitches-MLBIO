import argparse
import builtins
import gc
import importlib
import json
import signal
import sys
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import AutoTokenizer
from transformers.dynamic_module_utils import get_class_from_dynamic_module

SRC_ROOT = Path(__file__).resolve().parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data_engineering.paths import resolve_path
from validation_embedding.config import LLMSettings, load_llm_settings
from validation_embedding.storage import ArtifactStorage, get_storage

DEFAULT_INPUT = "data/processed/de_novo/smoke/generated.fasta"
BATCH_INPUT = "data/processed/de_novo/generated.fasta"
_BIRNA_CPU_PATCHED = False
_STORAGE = None
_EMBEDDING_SUBDIR = ""


def get_device(prefer_cuda: bool = True, require_cuda: bool = False) -> torch.device:
    """Return a CUDA device when available, otherwise CPU."""
    if prefer_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"BiRNA device: {device} ({torch.cuda.get_device_name(0)})")
        return device
    if require_cuda:
        raise RuntimeError(
            "CUDA is required for BiRNA embedding on this host, but torch.cuda.is_available() "
            "is False. Do not pip-install torch over the RunPod image PyTorch build."
        )
    print("BiRNA device: cpu (CUDA unavailable)")
    return torch.device("cpu")


def _patch_birna_cpu_compat() -> None:
    """Mac/CPU-only: skip Triton flash-attn imports that break without GPU kernels."""
    global _BIRNA_CPU_PATCHED
    if _BIRNA_CPU_PATCHED:
        return

    import transformers.dynamic_module_utils as dmu

    def check_imports(filename: str) -> list[str]:
        """Import modeling-file deps, skipping Triton on CPU."""
        missing = []
        for imp in dmu.get_imports(filename):
            if imp == "triton":
                continue
            try:
                importlib.import_module(imp)
            except ImportError as exception:
                if "No module named" in str(exception):
                    missing.append(imp)
                else:
                    raise
        if missing:
            raise ImportError(
                "This modeling file requires the following packages that were not found "
                f"in your environment: {', '.join(missing)}."
            )
        return dmu.get_relative_imports(filename)

    dmu.check_imports = check_imports
    real_import = builtins.__import__

    def guarded_import(
        name: str,
        globals: dict[str, object] | None = None,
        locals: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        """Raise ImportError for Triton flash-attn on CPU Bert layers."""
        module_name = globals.get("__name__", "") if globals else ""
        if (
            module_name.endswith("bert_layers")
            and fromlist
            and "flash_attn_qkvpacked_func" in fromlist
        ):
            raise ImportError("CPU smoke test: skip Triton flash attention")
        return real_import(name, globals, locals, fromlist, level)

    builtins.__import__ = guarded_import
    _BIRNA_CPU_PATCHED = True


def _patch_birna_alibi(module: object, device: torch.device) -> None:
    """Keep ALiBi tensors on the same device as the model (GPU when available)."""
    encoder = module.BertEncoder
    original_rebuild = encoder.rebuild_alibi_tensor
    default_device = torch.device(device)

    def rebuild_alibi(
        self: object, size: int, device: torch.device | None = None
    ) -> torch.Tensor:
        """Rebuild the ALiBi tensor on the model device by default."""
        if device is None:
            device = default_device
        return original_rebuild(self, size, device=device)

    encoder.rebuild_alibi_tensor = rebuild_alibi


def parse_fasta(path: Path | str) -> list[tuple[str, str]]:
    """Parse a FASTA file into (header, sequence) pairs."""
    records = []
    header = None
    parts = []
    with Path(path).open() as handle:
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


def nuc_tokenize_input(sequence: str) -> str:
    """Convert a DNA/RNA string into space-separated NUC tokens."""
    sequence = sequence.replace("T", "U").upper()
    return " ".join(list(sequence))


def load_birna_model(
    settings: LLMSettings, device: torch.device
) -> tuple[AutoTokenizer, torch.nn.Module, transformers.BertConfig]:
    """Load BiRNA-BERT, tokenizer, and config onto the given device."""
    # CPU-compat patches only when actually on CPU (Mac smoke). On CUDA, leave
    # Triton/flash-attn imports alone so the A100 path can engage.
    if device.type == "cpu":
        _patch_birna_cpu_compat()
    tokenizer = AutoTokenizer.from_pretrained(settings.birna_tokenizer_id)
    config = transformers.BertConfig.from_pretrained(settings.birna_model_id)
    model_cls = get_class_from_dynamic_module(
        "bert_layers.BertForMaskedLM",
        settings.birna_model_id,
    )
    _patch_birna_alibi(sys.modules[model_cls.__module__], device)
    # float32 on GPU: BiRNA's custom attention path mismatches float16 (Half vs Float).
    dtype = torch.float32
    model = model_cls.from_pretrained(
        settings.birna_model_id,
        config=config,
        torch_dtype=dtype,
    )
    model.cls = torch.nn.Identity()
    model.to(device)
    model.eval()
    print(f"BiRNA model on {next(model.parameters()).device}, dtype={dtype}")
    return tokenizer, model, config


def _record_id(header: str) -> str:
    """Sanitize a FASTA header into a filesystem-safe record id."""
    return header.replace("/", "_").replace(" ", "_")


def _sort_pending_by_length(
    pending: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    """Sort by sequence length so micro-batches share similar lengths (less pad waste)."""
    return sorted(pending, key=lambda item: len(item[1]))


def embed_sequence(
    tokenizer: AutoTokenizer,
    model: torch.nn.Module,
    config: transformers.BertConfig,
    sequence: str,
    device: torch.device,
) -> tuple[torch.Tensor, str]:
    """Embed a single sequence and return (tensor, NUC input)."""
    results = embed_batch(tokenizer, model, config, [sequence], device)
    return results[0]


def embed_batch(
    tokenizer: AutoTokenizer,
    model: torch.nn.Module,
    config: transformers.BertConfig,
    sequences: list[str],
    device: torch.device,
) -> list[tuple[torch.Tensor, str]]:
    """True GPU micro-batch: pad within the batch, one forward, slice off pad tokens."""
    if not sequences:
        return []
    nuc_inputs = []
    for sequence in sequences:
        nuc_input = nuc_tokenize_input(sequence)
        if " " not in nuc_input.strip():
            raise ValueError("NUC tokenization requires space-separated nucleotides")
        nuc_inputs.append(nuc_input)

    tokens = tokenizer(nuc_inputs, padding=True, return_tensors="pt")
    batch = {key: value.to(device) for key, value in tokens.items()}
    with torch.no_grad():
        output = model(**batch)
    embeddings = output.logits
    if embeddings.ndim != 3:
        raise ValueError(
            f"Expected 3D embedding tensor, got shape {tuple(embeddings.shape)}"
        )
    if embeddings.shape[-1] != config.hidden_size:
        raise ValueError(
            f"Hidden size mismatch: expected {config.hidden_size}, got {embeddings.shape[-1]}"
        )

    attention_mask = batch["attention_mask"]
    results = []
    for index, nuc_input in enumerate(nuc_inputs):
        length = int(attention_mask[index].sum().item())
        # Keep special tokens; drop pad positions only (matches single-seq path).
        row = embeddings[index, :length, :].unsqueeze(0).contiguous()
        results.append((row, nuc_input))
    return results


def save_embedding(
    record_id: str,
    sequence: str,
    embeddings: torch.Tensor,
    config: transformers.BertConfig,
    output_dir: Path | str,
    settings: LLMSettings,
    storage: ArtifactStorage,
) -> tuple[Path, Path]:
    """Write embedding .npy/.json files and append a manifest row."""
    output_dir = resolve_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    array = embeddings.cpu().numpy()
    npy_path = output_dir / f"{record_id}.npy"
    meta_path = output_dir / f"{record_id}.json"
    np.save(npy_path, array)
    metadata = {
        "record_id": record_id,
        "sequence": sequence,
        "tokenization": "NUC",
        "model_id": settings.birna_model_id,
        "shape": list(array.shape),
        "hidden_size": config.hidden_size,
    }
    with meta_path.open("w") as handle:
        json.dump(metadata, handle, indent=2)
    storage.append_jsonl(output_dir / "manifest.jsonl", metadata)
    return npy_path, meta_path


def _flush_storage() -> None:
    """Upload pending embedding artifacts if a storage client is registered."""
    global _STORAGE, _EMBEDDING_SUBDIR
    if _STORAGE is not None:
        _STORAGE.sync_up(embedding_subdir=_EMBEDDING_SUBDIR)


def _register_sigterm_handler(storage: ArtifactStorage, embedding_subdir: str) -> None:
    """Flush embedding artifacts to storage when the process receives SIGTERM."""
    global _STORAGE, _EMBEDDING_SUBDIR
    _STORAGE = storage
    _EMBEDDING_SUBDIR = embedding_subdir

    def _handle_sigterm(signum: int, frame: object | None) -> None:
        """Flush artifacts to storage and exit on SIGTERM."""
        print("SIGTERM received — flushing embedding artifacts before exit...")
        storage.write_run_state(status="preempted", event="sigterm_embed")
        _flush_storage()
        raise SystemExit(143)

    signal.signal(signal.SIGTERM, _handle_sigterm)


def run_birna_embed(
    input_fasta: str = DEFAULT_INPUT,
    output_subdir: str = "smoke",
    sequence: str | None = None,
    batch_size: int = 32,
    resume: bool = False,
    dry_run: bool = False,
    require_cuda: bool = False,
) -> list[tuple[Path, Path]]:
    """Embed sequences with BiRNA-BERT and write checkpointed .npy artifacts."""
    settings = load_llm_settings()
    storage = get_storage(settings)
    output_dir = resolve_path(settings.embedding_output_dir) / output_subdir
    # Cloud / RunPod targets must use the A100; local Mac may fall back to CPU.
    require_cuda = require_cuda or settings.storage_target in {"runpod", "s3", "gcs"}
    device = get_device(prefer_cuda=True, require_cuda=require_cuda)

    if resume:
        storage.sync_down(embedding_subdir=output_subdir)

    if dry_run:
        input_path = None if sequence else resolve_path(input_fasta)
        completed = storage.completed_embedding_ids(output_subdir) if resume else set()
        print("BiRNA-BERT dry-run")
        print(f"  input:     {input_path or 'inline'}")
        print(f"  output:    {output_dir}")
        print(f"  model:     {settings.birna_model_id}")
        print(f"  device:    {device}")
        print(f"  resume:    {resume} ({len(completed)} already embedded)")
        print(f"  batch_size:{batch_size} (length-bucketed GPU micro-batches)")
        return []

    if sequence:
        records = [("inline_0", sequence)]
        input_path = None
    else:
        input_path = resolve_path(input_fasta)
        records = parse_fasta(input_path)

    if not records:
        raise ValueError(f"No sequences found in {input_path}")

    completed_ids = storage.completed_embedding_ids(output_subdir) if resume else set()
    pending = [
        (header, seq)
        for header, seq in records
        if _record_id(header) not in completed_ids
    ]

    if not pending:
        print(f"Embedding complete: {len(completed_ids)} records already in manifest")
        return []

    pending = _sort_pending_by_length(pending)
    print(
        f"Length-bucketed batching: {len(pending)} pending, batch_size={batch_size}, "
        f"len_range=[{len(pending[0][1])}, {len(pending[-1][1])}]"
    )

    _register_sigterm_handler(storage, output_subdir)
    tokenizer, model, config = load_birna_model(settings, device)
    saved = []
    total_batches = (len(pending) + batch_size - 1) // batch_size

    for batch_idx, batch_start in enumerate(
        range(0, len(pending), batch_size), start=1
    ):
        batch = pending[batch_start : batch_start + batch_size]
        headers = [header for header, _ in batch]
        seqs = [seq for _, seq in batch]
        lengths = [len(seq) for seq in seqs]
        results = embed_batch(tokenizer, model, config, seqs, device)

        batch_ids = []
        for header, seq, (embeddings, nuc_input) in zip(headers, seqs, results):
            record_id = _record_id(header)
            paths = save_embedding(
                record_id, seq, embeddings, config, output_dir, settings, storage
            )
            saved.append(paths)
            batch_ids.append(record_id)

        print(
            f"batch {batch_idx}/{total_batches}: size={len(batch)}, "
            f"len_range=[{min(lengths)}, {max(lengths)}], device={device}"
        )

        storage.write_run_state(
            status="embedding",
            embedded=len(storage.completed_embedding_ids(output_subdir)),
        )
        # Upload only this batch's artifacts + manifest (not the full embedding tree).
        storage.upload_embedding_artifacts(output_subdir, record_ids=batch_ids)
        storage.upload_run_state()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        print(
            f"Embedding checkpoint: "
            f"{len(storage.completed_embedding_ids(output_subdir))} total records "
            f"on {device}"
        )

    print(f"Wrote embeddings to {output_dir}")
    storage.write_run_state(
        status="embedding_complete",
        embedded=len(storage.completed_embedding_ids(output_subdir)),
    )
    storage.sync_up(embedding_subdir=output_subdir)
    return saved


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for BiRNA-BERT embedding."""
    parser = argparse.ArgumentParser(description="BiRNA-BERT NUC embedding.")
    parser.add_argument("--input-fasta", default=None)
    parser.add_argument("--output-subdir", default=None)
    parser.add_argument(
        "--sequence", default=None, help="Inline sequence instead of FASTA"
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Fail if CUDA is unavailable (default on runpod/s3/gcs storage targets).",
    )
    return parser


def main() -> None:
    """Run BiRNA-BERT embedding from the command line."""
    from validation_embedding.runpod_lifecycle import runpod_terminate_if_configured

    args = _build_parser().parse_args()
    input_fasta = args.input_fasta or DEFAULT_INPUT
    output_subdir = args.output_subdir
    if output_subdir is None:
        output_subdir = "smoke" if "smoke" in input_fasta else ""
    try:
        run_birna_embed(
            input_fasta=input_fasta,
            output_subdir=output_subdir,
            sequence=args.sequence,
            batch_size=args.batch_size,
            resume=args.resume,
            dry_run=args.dry_run,
            require_cuda=args.require_cuda,
        )
    finally:
        # Standalone BiRNA jobs stop the pod; orchestrator sets RUNPOD_SKIP_TERMINATE.
        if not args.dry_run:
            runpod_terminate_if_configured()


if __name__ == "__main__":
    main()
