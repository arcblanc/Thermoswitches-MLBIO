import torch
from transformers import AutoTokenizer

from de_novo_hallucinations.genererna.loader import autocast_context
from de_novo_hallucinations.genererna.model import GPT


def generate_sequences(
    model: GPT,
    tokenizer: AutoTokenizer,
    device: torch.device,
    *,
    num_samples: int = 2,
    max_new_tokens: int = 64,
    strategy: str = "top_k",
    temperature: float = 1.0,
    top_k: int = 250,
    start: str = "<|endoftext|>",
    repetition_penalty: float = 1.0,
) -> list[str]:
    """Sample RNA sequences from a GenerRNA GPT checkpoint."""
    encode = tokenizer.encode
    decode = tokenizer.decode
    start_ids = encode("".join(start))
    prompt = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]
    sequences = []

    with torch.no_grad():
        with autocast_context(device):
            for _ in range(num_samples):
                token_sequence = model.generate(
                    prompt,
                    max_new_tokens,
                    strategy=strategy,
                    temperature=temperature,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                )[0].tolist()
                text = decode(token_sequence).replace(" ", "").upper()
                sequences.append(text)

    return sequences
