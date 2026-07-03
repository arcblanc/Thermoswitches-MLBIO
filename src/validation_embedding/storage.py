import json
from pathlib import Path

from data_engineering.paths import resolve_path
from validation_embedding.config import LLMSettings, load_llm_settings

DE_NOVO_REMOTE_SUBDIR = "de_novo"
EMBEDDING_REMOTE_SUBDIR = "validation_embedding"
RUN_STATE_NAME = "run_state.json"
GENERATION_MANIFEST_NAME = "generation_manifest.jsonl"
EMBEDDING_MANIFEST_NAME = "manifest.jsonl"
GENERATED_FASTA_NAME = "generated.fasta"


class ArtifactStorage:
    def __init__(self, settings: LLMSettings | None = None):
        self.settings = settings or load_llm_settings()
        self._gcs_client = None
        self._s3_client = None

    @property
    def uses_gcs(self) -> bool:
        return self.settings.storage_target == "gcs"

    @property
    def uses_s3(self) -> bool:
        return self.settings.storage_target in {"s3", "runpod"}

    @property
    def uses_remote(self) -> bool:
        return self.uses_gcs or self.uses_s3

    def _remote_key(self, relative_path: str) -> str:
        relative_path = relative_path.lstrip("/")
        if self.uses_gcs:
            prefix = self.settings.gcs_prefix
        else:
            prefix = self.settings.aws_s3_prefix
        return f"{prefix}/{relative_path}" if prefix else relative_path

    def _require_gcs(self):
        if not self.settings.gcs_bucket:
            raise ValueError("GCS_BUCKET is required when STORAGE_TARGET=gcs")

    def _require_s3(self):
        if not self.settings.aws_s3_bucket:
            raise ValueError("AWS_S3_BUCKET is required when STORAGE_TARGET=s3 or runpod")

    def _gcs_client_or_raise(self):
        self._require_gcs()
        if self._gcs_client is None:
            from google.cloud import storage

            self._gcs_client = storage.Client()
        return self._gcs_client

    def _s3_client_or_raise(self):
        self._require_s3()
        if self._s3_client is None:
            import boto3

            self._s3_client = boto3.client("s3", region_name=self.settings.aws_region)
        return self._s3_client

    def _gcs_bucket(self):
        return self._gcs_client_or_raise().bucket(self.settings.gcs_bucket)

    def local_path(self, relative_path: str) -> Path:
        return resolve_path(relative_path)

    def de_novo_fasta_path(self) -> Path:
        return self.local_path(
            f"{self.settings.de_novo_output_dir}/{GENERATED_FASTA_NAME}"
        )

    def generation_manifest_path(self) -> Path:
        return self.local_path(
            f"{self.settings.de_novo_output_dir}/{GENERATION_MANIFEST_NAME}"
        )

    def embedding_dir(self, subdir: str = "") -> Path:
        base = resolve_path(self.settings.embedding_output_dir)
        return base / subdir if subdir else base

    def embedding_manifest_path(self, subdir: str = "") -> Path:
        return self.embedding_dir(subdir) / EMBEDDING_MANIFEST_NAME

    def run_state_path(self) -> Path:
        return self.local_path(f"data/processed/{RUN_STATE_NAME}")

    def upload_file(self, local_path: Path, remote_relative: str | None = None) -> None:
        if not self.uses_remote:
            return
        local_path = Path(local_path)
        if not local_path.exists():
            return
        remote_relative = remote_relative or str(local_path.relative_to(resolve_path(".")))
        remote_key = self._remote_key(remote_relative)
        if self.uses_gcs:
            blob = self._gcs_bucket().blob(remote_key)
            blob.upload_from_filename(str(local_path))
        else:
            self._s3_client_or_raise().upload_file(
                str(local_path),
                self.settings.aws_s3_bucket,
                remote_key,
            )

    def download_file(self, remote_relative: str, local_path: Path) -> bool:
        if not self.uses_remote:
            return local_path.exists()
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        remote_key = self._remote_key(remote_relative)
        if self.uses_gcs:
            blob = self._gcs_bucket().blob(remote_key)
            if not blob.exists():
                return False
            blob.download_to_filename(str(local_path))
        else:
            client = self._s3_client_or_raise()
            from botocore.exceptions import ClientError

            try:
                client.head_object(Bucket=self.settings.aws_s3_bucket, Key=remote_key)
            except ClientError:
                return False
            client.download_file(self.settings.aws_s3_bucket, remote_key, str(local_path))
        return True

    def _list_remote_objects(self, prefix: str) -> list[str]:
        names = []
        remote_prefix = self._remote_key(prefix)
        if self.uses_gcs:
            bucket = self._gcs_bucket()
            blob_prefix = remote_prefix if remote_prefix.endswith("/") else f"{remote_prefix}/"
            for blob in bucket.list_blobs(prefix=blob_prefix):
                name = blob.name[len(blob_prefix) :]
                if name and "/" not in name:
                    names.append(name)
        else:
            client = self._s3_client_or_raise()
            from botocore.exceptions import ClientError

            paginator = client.get_paginator("list_objects_v2")
            try:
                for page in paginator.paginate(
                    Bucket=self.settings.aws_s3_bucket,
                    Prefix=remote_prefix.rstrip("/") + "/",
                ):
                    for obj in page.get("Contents", []):
                        key = obj["Key"]
                        name = key.split("/")[-1]
                        if name:
                            names.append(name)
            except ClientError as exc:
                code = exc.response.get("Error", {}).get("Code", "")
                if code not in {"AccessDenied", "403"}:
                    raise
        return names

    def upload_de_novo_artifacts(self) -> None:
        if not self.uses_remote:
            return
        fasta = self.de_novo_fasta_path()
        manifest = self.generation_manifest_path()
        if fasta.exists():
            self.upload_file(
                fasta,
                f"{DE_NOVO_REMOTE_SUBDIR}/{GENERATED_FASTA_NAME}",
            )
        if manifest.exists():
            self.upload_file(
                manifest,
                f"{DE_NOVO_REMOTE_SUBDIR}/{GENERATION_MANIFEST_NAME}",
            )

    def upload_embedding_artifacts(self, subdir: str = "") -> None:
        if not self.uses_remote:
            return
        output_dir = self.embedding_dir(subdir)
        if not output_dir.exists():
            return
        prefix = EMBEDDING_REMOTE_SUBDIR
        if subdir:
            prefix = f"{prefix}/{subdir}"
        manifest = output_dir / EMBEDDING_MANIFEST_NAME
        if manifest.exists():
            self.upload_file(manifest, f"{prefix}/{EMBEDDING_MANIFEST_NAME}")
        for path in output_dir.glob("*.npy"):
            self.upload_file(path, f"{prefix}/{path.name}")
        for path in output_dir.glob("*.json"):
            if path.name == EMBEDDING_MANIFEST_NAME:
                continue
            self.upload_file(path, f"{prefix}/{path.name}")

    def upload_run_state(self) -> None:
        path = self.run_state_path()
        if path.exists():
            self.upload_file(path, RUN_STATE_NAME)

    def sync_up(self, embedding_subdir: str = "") -> None:
        self.upload_de_novo_artifacts()
        self.upload_embedding_artifacts(embedding_subdir)
        self.upload_run_state()

    def sync_down(self, embedding_subdir: str = "") -> None:
        if not self.uses_remote:
            return
        self.download_file(
            f"{DE_NOVO_REMOTE_SUBDIR}/{GENERATED_FASTA_NAME}",
            self.de_novo_fasta_path(),
        )
        self.download_file(
            f"{DE_NOVO_REMOTE_SUBDIR}/{GENERATION_MANIFEST_NAME}",
            self.generation_manifest_path(),
        )
        prefix = EMBEDDING_REMOTE_SUBDIR
        if embedding_subdir:
            prefix = f"{prefix}/{embedding_subdir}"
        output_dir = self.embedding_dir(embedding_subdir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.download_file(
            f"{prefix}/{EMBEDDING_MANIFEST_NAME}",
            self.embedding_manifest_path(embedding_subdir),
        )
        for name in self._list_remote_objects(prefix):
            if name.endswith((".npy", ".json")) and name != EMBEDDING_MANIFEST_NAME:
                self.download_file(f"{prefix}/{name}", output_dir / name)
        self.download_file(RUN_STATE_NAME, self.run_state_path())

    def load_jsonl(self, path: Path) -> list[dict]:
        if not path.exists():
            return []
        rows = []
        with path.open() as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def append_jsonl(self, path: Path, record: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")

    def load_generation_manifest(self) -> list[dict]:
        return self.load_jsonl(self.generation_manifest_path())

    def load_embedding_manifest(self, subdir: str = "") -> list[dict]:
        return self.load_jsonl(self.embedding_manifest_path(subdir))

    def completed_generation_ids(self) -> set[str]:
        return {row["record_id"] for row in self.load_generation_manifest()}

    def completed_embedding_ids(self, subdir: str = "") -> set[str]:
        return {row["record_id"] for row in self.load_embedding_manifest(subdir)}

    def write_run_state(self, **fields) -> dict:
        path = self.run_state_path()
        state = {}
        if path.exists():
            state = json.loads(path.read_text())
        state.update(fields)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state, indent=2))
        self.upload_run_state()
        return state


def get_storage(settings: LLMSettings | None = None) -> ArtifactStorage:
    return ArtifactStorage(settings=settings)
