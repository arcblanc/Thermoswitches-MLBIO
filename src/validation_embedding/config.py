import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from data_engineering.paths import PROJECT_ROOT

load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class LLMSettings:
    hf_token: str | None
    generna_cache_dir: str
    birna_tokenizer_id: str
    birna_model_id: str
    embedding_output_dir: str
    de_novo_output_dir: str


def load_llm_settings() -> LLMSettings:
    return LLMSettings(
        hf_token=os.environ.get("HF_TOKEN") or None,
        generna_cache_dir=os.environ.get("GENERNA_CACHE_DIR", "models/genererna"),
        birna_tokenizer_id=os.environ.get("BIRNA_TOKENIZER_ID", "buetnlpbio/birna-tokenizer"),
        birna_model_id=os.environ.get("BIRNA_MODEL_ID", "buetnlpbio/birna-bert"),
        embedding_output_dir=os.environ.get(
            "EMBEDDING_OUTPUT_DIR", "data/processed/validation_embedding"
        ),
        de_novo_output_dir=os.environ.get("DE_NOVO_OUTPUT_DIR", "data/processed/de_novo"),
    )
