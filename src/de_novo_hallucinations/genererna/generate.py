import torch

from de_novo_hallucinations.genererna.loader import autocast_context


def generate_sequences(
    model,
    tokenizer,
    device,
    *,
    num_samples=2,
    max_new_tokens=64,
    strategy="top_k",
    temperature=1.0,
    top_k=250,
    start="<|endoftext|>",
    repetition_penalty=1.0,
):
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
