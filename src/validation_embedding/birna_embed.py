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
from validation_embedding.config import load_llm_settings
from validation_embedding.storage import get_storage

DEFAULT_INPUT = "data/processed/de_novo/smoke/generated.fasta"
BATCH_INPUT = "data/processed/de_novo/generated.fasta"
_BIRNA_CPU_PATCHED = False
_STORAGE = None
_EMBEDDING_SUBDIR = ""


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _patch_birna_cpu_compat():
    global _BIRNA_CPU_PATCHED
    if _BIRNA_CPU_PATCHED:
        return

    import transformers.dynamic_module_utils as dmu

    def check_imports(filename):
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

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
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


def _patch_birna_alibi(module):
    encoder = module.BertEncoder
    original_rebuild = encoder.rebuild_alibi_tensor

    def rebuild_alibi(self, size, device=None):
        if device is None:
            device = torch.device("cpu")
        return original_rebuild(self, size, device=device)

    encoder.rebuild_alibi_tensor = rebuild_alibi


def parse_fasta(path):
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


def nuc_tokenize_input(sequence):
    sequence = sequence.replace("T", "U").upper()
    return " ".join(list(sequence))


def load_birna_model(settings, device):
    _patch_birna_cpu_compat()
    tokenizer = AutoTokenizer.from_pretrained(settings.birna_tokenizer_id)
    config = transformers.BertConfig.from_pretrained(settings.birna_model_id)
    model_cls = get_class_from_dynamic_module(
        "bert_layers.BertForMaskedLM",
        settings.birna_model_id,
    )
    _patch_birna_alibi(sys.modules[model_cls.__module__])
    model = model_cls.from_pretrained(
        settings.birna_model_id,
        config=config,
        torch_dtype=torch.float32,
    )
    model.cls = torch.nn.Identity()
    model.to(device)
    model.eval()
    return tokenizer, model, config


def embed_sequence(tokenizer, model, config, sequence, device):
    nuc_input = nuc_tokenize_input(sequence)
    if " " not in nuc_input.strip():
        raise ValueError("NUC tokenization requires space-separated nucleotides")
    tokens = tokenizer(nuc_input, return_tensors="pt")
    batch = {key: value.to(device) for key, value in tokens.items()}
    with torch.no_grad():
        output = model(**batch)
    embeddings = output.logits
    if embeddings.ndim != 3:
        raise ValueError(f"Expected 3D embedding tensor, got shape {tuple(embeddings.shape)}")
    if embeddings.shape[-1] != config.hidden_size:
        raise ValueError(
            f"Hidden size mismatch: expected {config.hidden_size}, got {embeddings.shape[-1]}"
        )
    return embeddings, nuc_input


def save_embedding(record_id, sequence, embeddings, config, output_dir, settings, storage):
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


def _flush_storage():
    global _STORAGE, _EMBEDDING_SUBDIR
    if _STORAGE is not None:
        _STORAGE.sync_up(embedding_subdir=_EMBEDDING_SUBDIR)


def _register_sigterm_handler(storage, embedding_subdir):
    global _STORAGE, _EMBEDDING_SUBDIR
    _STORAGE = storage
    _EMBEDDING_SUBDIR = embedding_subdir

    def _handle_sigterm(signum, frame):
        print("SIGTERM received — flushing embedding artifacts before exit...")
        storage.write_run_state(status="preempted", event="sigterm_embed")
        _flush_storage()
        raise SystemExit(143)

    signal.signal(signal.SIGTERM, _handle_sigterm)


def run_birna_embed(
    input_fasta=DEFAULT_INPUT,
    output_subdir="smoke",
    sequence=None,
    batch_size=32,
    resume=False,
    dry_run=False,
):
    settings = load_llm_settings()
    storage = get_storage(settings)
    output_dir = resolve_path(settings.embedding_output_dir) / output_subdir
    device = get_device()

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
        print(f"  batch_size:{batch_size}")
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
        if header.replace("/", "_").replace(" ", "_") not in completed_ids
    ]

    if not pending:
        print(f"Embedding complete: {len(completed_ids)} records already in manifest")
        return []

    _register_sigterm_handler(storage, output_subdir)
    tokenizer, model, config = load_birna_model(settings, device)
    saved = []

    for batch_start in range(0, len(pending), batch_size):
        batch = pending[batch_start : batch_start + batch_size]
        for header, seq in batch:
            record_id = header.replace("/", "_").replace(" ", "_")
            embeddings, nuc_input = embed_sequence(tokenizer, model, config, seq, device)
            print(
                f"{record_id}: NUC input length={len(nuc_input.split())}, "
                f"embedding shape={tuple(embeddings.shape)}"
            )
            paths = save_embedding(
                record_id, seq, embeddings, config, output_dir, settings, storage
            )
            saved.append(paths)

        storage.write_run_state(
            status="embedding",
            embedded=len(storage.completed_embedding_ids(output_subdir)),
        )
        storage.sync_up(embedding_subdir=output_subdir)
        gc.collect()
        print(
            f"Embedding checkpoint: "
            f"{len(storage.completed_embedding_ids(output_subdir))} total records"
        )

    print(f"Wrote embeddings to {output_dir}")
    storage.write_run_state(
        status="embedding_complete",
        embedded=len(storage.completed_embedding_ids(output_subdir)),
    )
    storage.sync_up(embedding_subdir=output_subdir)
    return saved


def _build_parser():
    parser = argparse.ArgumentParser(description="BiRNA-BERT NUC embedding.")
    parser.add_argument("--input-fasta", default=None)
    parser.add_argument("--output-subdir", default=None)
    parser.add_argument("--sequence", default=None, help="Inline sequence instead of FASTA")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main():
    args = _build_parser().parse_args()
    input_fasta = args.input_fasta or DEFAULT_INPUT
    output_subdir = args.output_subdir
    if output_subdir is None:
        output_subdir = "smoke" if "smoke" in input_fasta else ""
    run_birna_embed(
        input_fasta=input_fasta,
        output_subdir=output_subdir,
        sequence=args.sequence,
        batch_size=args.batch_size,
        resume=args.resume,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
