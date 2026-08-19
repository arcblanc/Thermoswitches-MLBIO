import os
from dataclasses import dataclass

from dotenv import load_dotenv

from data_engineering.paths import PROJECT_ROOT

load_dotenv(PROJECT_ROOT / ".env")


def _env_bool(name: str, default: bool = False) -> bool:
    """Parse a boolean environment variable, falling back to default."""
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
    aws_s3_bucket: str | None
    aws_region: str
    aws_s3_prefix: str
    runpod_pod_id: str | None
    runpod_api_key: str | None
    runpod_auto_terminate: bool
    vm_auto_shutdown: bool
    generna_batch_size: int
    generna_num_samples: int
    # EVA (Option B TaxID panel + chunk gates)
    eva_checkpoint_dir: str
    eva_cache_dir: str
    eva_chunk_size: int
    eva_min_len: int
    eva_max_len: int
    eva_rna_type: str
    eva_num_samples: int


def load_llm_settings() -> LLMSettings:
    """Load LLM and storage settings from environment variables."""
    return LLMSettings(
        hf_token=os.environ.get("HF_TOKEN") or None,
        generna_cache_dir=os.environ.get("GENERNA_CACHE_DIR", "models/genererna"),
        birna_tokenizer_id=os.environ.get(
            "BIRNA_TOKENIZER_ID", "buetnlpbio/birna-tokenizer"
        ),
        birna_model_id=os.environ.get("BIRNA_MODEL_ID", "buetnlpbio/birna-bert"),
        embedding_output_dir=os.environ.get(
            "EMBEDDING_OUTPUT_DIR", "data/processed/validation_embedding"
        ),
        de_novo_output_dir=os.environ.get(
            "DE_NOVO_OUTPUT_DIR", "data/processed/de_novo"
        ),
        storage_target=os.environ.get("STORAGE_TARGET", "local").strip().lower(),
        gcs_bucket=os.environ.get("GCS_BUCKET") or None,
        gcs_prefix=os.environ.get("GCS_PREFIX", "llm-batch/v1").strip("/"),
        aws_s3_bucket=os.environ.get("AWS_S3_BUCKET") or None,
        aws_region=os.environ.get("AWS_REGION", "us-east-1"),
        aws_s3_prefix=os.environ.get("AWS_S3_PREFIX", "llm-batch/v1").strip("/"),
        runpod_pod_id=os.environ.get("RUNPOD_POD_ID") or None,
        runpod_api_key=os.environ.get("RUNPOD_API_KEY") or None,
        runpod_auto_terminate=_env_bool("RUNPOD_AUTO_TERMINATE", default=False),
        vm_auto_shutdown=_env_bool("VM_AUTO_SHUTDOWN", default=False),
        generna_batch_size=int(os.environ.get("GENERNA_BATCH_SIZE", "50")),
        generna_num_samples=int(os.environ.get("GENERNA_NUM_SAMPLES", "10000")),
        eva_checkpoint_dir=os.environ.get(
            "EVA_CHECKPOINT_DIR", "models/eva/checkpoint"
        ),
        eva_cache_dir=os.environ.get("EVA_CACHE_DIR", "models/eva"),
        eva_chunk_size=int(os.environ.get("EVA_CHUNK_SIZE", "512")),
        eva_min_len=int(os.environ.get("EVA_MIN_LEN", "40")),
        eva_max_len=int(os.environ.get("EVA_MAX_LEN", "600")),
        eva_rna_type=os.environ.get("EVA_RNA_TYPE", "mRNA"),
        eva_num_samples=int(os.environ.get("EVA_NUM_SAMPLES", "16")),
    )
