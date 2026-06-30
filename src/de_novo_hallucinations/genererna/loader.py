from contextlib import nullcontext

import torch
from transformers import AutoTokenizer

from de_novo_hallucinations.genererna.model import GPT, GPTConfig


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_tokenizer(tokenizer_path):
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
    probe = "AUGC"
    ids = tokenizer.encode(probe)
    decoded = tokenizer.decode(ids).replace(" ", "").upper()
    if decoded != probe:
        raise ValueError(f"Tokenizer round-trip failed: {probe!r} -> {decoded!r}")
    return tokenizer


def load_generna_model(ckpt_path, device=None):
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


def autocast_context(device):
    if device.type == "cuda":
        return torch.amp.autocast(device_type="cuda", dtype=torch.float32)
    return nullcontext()
