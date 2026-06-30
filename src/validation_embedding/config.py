import os
from dataclasses import dataclass

from dotenv import load_dotenv

from data_engineering.paths import PROJECT_ROOT

load_dotenv(PROJECT_ROOT / ".env")


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class LLMSettings:
    hf_token: str | None
    generna_cache_dir: str
    birna_tokenizer_id: str
    birna_model_id: str
    embedding_output_dir: str
    de_novo_output_dir: str
    storage_target: str
    gcs_bucket: str | None
    gcs_prefix: str
    vm_auto_shutdown: bool
    generna_batch_size: int
    generna_num_samples: int


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
        storage_target=os.environ.get("STORAGE_TARGET", "local").strip().lower(),
        gcs_bucket=os.environ.get("GCS_BUCKET") or None,
        gcs_prefix=os.environ.get("GCS_PREFIX", "llm-batch/v1").strip("/"),
        vm_auto_shutdown=_env_bool("VM_AUTO_SHUTDOWN", default=False),
        generna_batch_size=int(os.environ.get("GENERNA_BATCH_SIZE", "50")),
        generna_num_samples=int(os.environ.get("GENERNA_NUM_SAMPLES", "10000")),
    )
