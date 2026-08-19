import os
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

GENERNA_REPO = "pfnet/GenerRNA"


def ensure_generna_assets(
    cache_dir: Path, hf_token: str | None = None
) -> dict[str, Path]:
    """Download checkpoint and BPE tokenizer if missing."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_dir = cache_dir / "tokenizer"
    ckpt_path = cache_dir / "model_updated.pt"

    token = hf_token or os.environ.get("HF_TOKEN") or None
    kwargs = {"token": token} if token else {}

    if not ckpt_path.exists():
        downloaded = hf_hub_download(
            repo_id=GENERNA_REPO,
            filename="model_updated.pt",
            local_dir=str(cache_dir),
            local_dir_use_symlinks=False,
            **kwargs,
        )
        ckpt_path = Path(downloaded)

    if not tokenizer_dir.exists() or not any(tokenizer_dir.iterdir()):
        snapshot_download(
            repo_id=GENERNA_REPO,
            allow_patterns=["tokenizer/*"],
            local_dir=str(cache_dir),
            local_dir_use_symlinks=False,
            **kwargs,
        )

    return {"ckpt_path": ckpt_path, "tokenizer_path": tokenizer_dir}
