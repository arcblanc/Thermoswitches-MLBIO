from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import cast

import torch
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from de_novo_hallucinations.genererna.model import GPT, GPTConfig


def get_device() -> torch.device:
    """Return CUDA when available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_tokenizer(tokenizer_path: Path | str) -> PreTrainedTokenizerBase:
    """Load a GenerRNA tokenizer and verify AUGC round-trip encoding."""
    tokenizer = cast(
        PreTrainedTokenizerBase, AutoTokenizer.from_pretrained(str(tokenizer_path))
    )
    probe = "AUGC"
    ids = tokenizer.encode(probe)
    decoded_text = tokenizer.decode(ids)
    if not isinstance(decoded_text, str):
        raise ValueError(
            f"Tokenizer decode returned non-string: {type(decoded_text)!r}"
        )
    decoded = decoded_text.replace(" ", "").upper()
    if decoded != probe:
        raise ValueError(f"Tokenizer round-trip failed: {probe!r} -> {decoded!r}")
    return tokenizer


def load_generna_model(
    ckpt_path: Path | str, device: torch.device | None = None
) -> tuple[GPT, torch.device]:
    """Load a GenerRNA GPT checkpoint and move it to the given device."""
    device = device or get_device()
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    gptconf = GPTConfig(**checkpoint["model_args"])
    model = GPT(gptconf)
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for key, value in list(state_dict.items()):
        if key.startswith(unwanted_prefix):
            state_dict[key[len(unwanted_prefix) :]] = state_dict.pop(key)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    return model, device


def autocast_context(device: torch.device) -> AbstractContextManager[object]:
    """Return a CUDA autocast context, or a no-op context on CPU."""
    if device.type == "cuda":
        return torch.amp.autocast(device_type="cuda", dtype=torch.float32)
    return nullcontext()
