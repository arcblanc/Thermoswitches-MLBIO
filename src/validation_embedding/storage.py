import json
from collections.abc import Iterator
from pathlib import Path
from typing import Protocol, cast

from data_engineering.paths import resolve_path
from validation_embedding.config import LLMSettings, load_llm_settings

DE_NOVO_REMOTE_SUBDIR = "de_novo"
EMBEDDING_REMOTE_SUBDIR = "validation_embedding"
RUN_STATE_NAME = "run_state.json"
GENERATION_MANIFEST_NAME = "generation_manifest.jsonl"
EMBEDDING_MANIFEST_NAME = "manifest.jsonl"
GENERATED_FASTA_NAME = "generated.fasta"


class GCSBlob(Protocol):
    name: str

    def upload_from_filename(self, filename: str) -> None: ...

    def download_to_filename(self, filename: str) -> None: ...

    def exists(self) -> bool: ...


class GCSBucket(Protocol):
    def blob(self, blob_name: str) -> GCSBlob: ...

    def list_blobs(self, *, prefix: str) -> Iterator[GCSBlob]: ...


class GCSClient(Protocol):
    def bucket(self, bucket_name: str) -> GCSBucket: ...


class S3Paginator(Protocol):
    def paginate(self, *, Bucket: str, Prefix: str) -> Iterator[dict[str, object]]: ...


class S3Client(Protocol):
    def upload_file(self, Filename: str, Bucket: str, Key: str) -> None: ...

    def head_object(self, *, Bucket: str, Key: str) -> object: ...

    def download_file(self, Bucket: str, Key: str, Filename: str) -> None: ...

    def get_paginator(self, operation_name: str) -> S3Paginator: ...


class ArtifactStorage:
    def __init__(self, settings: LLMSettings | None = None) -> None:
        """Initialize local and remote artifact storage from LLM settings."""
        self.settings = settings or load_llm_settings()
        self._gcs_client: GCSClient | None = None
        self._s3_client: S3Client | None = None

    @property
    def uses_gcs(self) -> bool:
        """Return whether storage uploads go to GCS."""
        return self.settings.storage_target == "gcs"

    @property
    def uses_s3(self) -> bool:
        """Return whether storage uploads go to S3 or RunPod S3."""
        return self.settings.storage_target in {"s3", "runpod"}

    @property
    def uses_remote(self) -> bool:
        """Return whether a remote object store is configured."""
        return self.uses_gcs or self.uses_s3

    def _remote_key(self, relative_path: str) -> str:
        """Prefix a relative path with the configured bucket prefix."""
        relative_path = relative_path.lstrip("/")
        if self.uses_gcs:
            prefix = self.settings.gcs_prefix
        else:
            prefix = self.settings.aws_s3_prefix
        return f"{prefix}/{relative_path}" if prefix else relative_path

    def _require_gcs(self) -> None:
        """Raise if GCS is selected but GCS_BUCKET is unset."""
        if not self.settings.gcs_bucket:
            raise ValueError("GCS_BUCKET is required when STORAGE_TARGET=gcs")

    def _require_s3(self) -> None:
        """Raise if S3 is selected but AWS_S3_BUCKET is unset."""
        if not self.settings.aws_s3_bucket:
            raise ValueError(
                "AWS_S3_BUCKET is required when STORAGE_TARGET=s3 or runpod"
            )

    def _gcs_client_or_raise(self) -> GCSClient:
        """Return a cached GCS client, creating it on first use."""
        self._require_gcs()
        if self._gcs_client is None:
            from google.cloud import storage

            self._gcs_client = cast(GCSClient, storage.Client())
        return self._gcs_client

    def _s3_client_or_raise(self) -> S3Client:
        """Return a cached boto3 S3 client, creating it on first use."""
        self._require_s3()
        if self._s3_client is None:
            import boto3

            self._s3_client = cast(
                S3Client,
                boto3.client("s3", region_name=self.settings.aws_region),
            )
        return self._s3_client

    def _gcs_bucket(self) -> GCSBucket:
        """Return the configured GCS bucket handle."""
        bucket_name = self.settings.gcs_bucket
        if bucket_name is None:
            raise ValueError("GCS_BUCKET is required when STORAGE_TARGET=gcs")
        return self._gcs_client_or_raise().bucket(bucket_name)

    def _s3_bucket_name(self) -> str:
        """Return the configured S3 bucket name."""
        bucket_name = self.settings.aws_s3_bucket
        if bucket_name is None:
            raise ValueError(
                "AWS_S3_BUCKET is required when STORAGE_TARGET=s3 or runpod"
            )
        return bucket_name

    def local_path(self, relative_path: str) -> Path:
        """Resolve a repo-relative path to an absolute local path."""
        return resolve_path(relative_path)

    def de_novo_fasta_path(self) -> Path:
        """Return the local path of the generated de novo FASTA."""
        return self.local_path(
            f"{self.settings.de_novo_output_dir}/{GENERATED_FASTA_NAME}"
        )

    def generation_manifest_path(self) -> Path:
        """Return the local path of the generation manifest JSONL."""
        return self.local_path(
            f"{self.settings.de_novo_output_dir}/{GENERATION_MANIFEST_NAME}"
        )

    def embedding_dir(self, subdir: str = "") -> Path:
        """Return the local embedding output directory, optionally nested."""
        base = resolve_path(self.settings.embedding_output_dir)
        return base / subdir if subdir else base

    def embedding_manifest_path(self, subdir: str = "") -> Path:
        """Return the local path of the embedding manifest JSONL."""
        return self.embedding_dir(subdir) / EMBEDDING_MANIFEST_NAME

    def run_state_path(self) -> Path:
        """Return the local path of the run_state.json file."""
        return self.local_path(f"data/processed/{RUN_STATE_NAME}")

    def upload_file(self, local_path: Path, remote_relative: str | None = None) -> bool:
        """Upload a local file to remote storage when a remote target is set."""
        if not self.uses_remote:
            return True
        local_path = Path(local_path)
        if not local_path.exists():
            return False
        remote_relative = remote_relative or str(
            local_path.relative_to(resolve_path("."))
        )
        remote_key = self._remote_key(remote_relative)
        try:
            if self.uses_gcs:
                blob = self._gcs_bucket().blob(remote_key)
                blob.upload_from_filename(str(local_path))
            else:
                self._s3_client_or_raise().upload_file(
                    str(local_path),
                    self._s3_bucket_name(),
                    remote_key,
                )
            return True
        except Exception as exc:
            print(f"WARNING: upload failed for {remote_relative}: {exc}")
            return False

    def download_file(self, remote_relative: str, local_path: Path) -> bool:
        """Download a remote object, skipping if a non-empty local file exists."""
        if not self.uses_remote:
            return local_path.exists()
        local_path = Path(local_path)
        # Resume-friendly: keep existing local artifacts (avoid re-pulling hundreds of .npy files).
        if local_path.exists() and local_path.stat().st_size > 0:
            return True
        local_path.parent.mkdir(parents=True, exist_ok=True)
        remote_key = self._remote_key(remote_relative)
        if self.uses_gcs:
            blob = self._gcs_bucket().blob(remote_key)
            if not blob.exists():
                return False
            blob.download_to_filename(str(local_path))
        else:
            client = self._s3_client_or_raise()
            bucket_name = self._s3_bucket_name()
            from botocore.exceptions import ClientError

            try:
                client.head_object(Bucket=bucket_name, Key=remote_key)
            except ClientError:
                return False
            client.download_file(bucket_name, remote_key, str(local_path))
        return True

    def _list_remote_objects(self, prefix: str) -> list[str]:
        """List object names under a remote prefix (non-recursive)."""
        names = []
        remote_prefix = self._remote_key(prefix)
        if self.uses_gcs:
            bucket = self._gcs_bucket()
            blob_prefix = (
                remote_prefix if remote_prefix.endswith("/") else f"{remote_prefix}/"
            )
            for blob in bucket.list_blobs(prefix=blob_prefix):
                name = blob.name[len(blob_prefix) :]
                if name and "/" not in name:
                    names.append(name)
        else:
            client = self._s3_client_or_raise()
            from botocore.exceptions import ClientError

            paginator = client.get_paginator("list_objects_v2")
            bucket_name = self._s3_bucket_name()
            try:
                for page in paginator.paginate(
                    Bucket=bucket_name,
                    Prefix=remote_prefix.rstrip("/") + "/",
                ):
                    contents = page.get("Contents", [])
                    if not isinstance(contents, list):
                        continue
                    for obj in contents:
                        if not isinstance(obj, dict):
                            continue
                        key = obj.get("Key")
                        if not isinstance(key, str):
                            continue
                        name = key.split("/")[-1]
                        if name:
                            names.append(name)
            except ClientError as exc:
                code = exc.response.get("Error", {}).get("Code", "")
                if code not in {"AccessDenied", "403"}:
                    raise
        return names

    def upload_de_novo_artifacts(self) -> None:
        """Upload generated FASTA, manifest, and accepted-chunk files."""
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
        # Immutable accepted slices for stream triage (S3-native watcher).
        accepted_dir = self.de_novo_fasta_path().parent / "accepted_chunks"
        if accepted_dir.is_dir():
            for path in sorted(accepted_dir.glob("accepted_*.fasta")):
                self.upload_file(
                    path,
                    f"{DE_NOVO_REMOTE_SUBDIR}/accepted_chunks/{path.name}",
                )

    def upload_embedding_artifacts(
        self, subdir: str = "", record_ids: list[str] | None = None
    ) -> None:
        """Upload embedding manifest and npy/json artifacts for a subdir."""
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
        # Full sync when record_ids is None; otherwise only upload this batch
        # (avoids re-uploading hundreds of .npy files every checkpoint).
        if record_ids is None:
            paths = list(output_dir.glob("*.npy")) + [
                path
                for path in output_dir.glob("*.json")
                if path.name != EMBEDDING_MANIFEST_NAME
            ]
        else:
            paths = []
            for record_id in record_ids:
                npy_path = output_dir / f"{record_id}.npy"
                json_path = output_dir / f"{record_id}.json"
                if npy_path.exists():
                    paths.append(npy_path)
                if json_path.exists():
                    paths.append(json_path)
        for path in paths:
            self.upload_file(path, f"{prefix}/{path.name}")

    def upload_run_state(self) -> None:
        """Upload run_state.json when it exists locally."""
        path = self.run_state_path()
        if path.exists():
            self.upload_file(path, RUN_STATE_NAME)

    def sync_up(self, embedding_subdir: str = "") -> None:
        """Upload de novo, embedding, and run-state artifacts."""
        self.upload_de_novo_artifacts()
        self.upload_embedding_artifacts(embedding_subdir)
        self.upload_run_state()

    def sync_down(self, embedding_subdir: str = "") -> None:
        """Download remote de novo, embedding, and run-state artifacts."""
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

    def load_jsonl(self, path: Path) -> list[dict[str, object]]:
        """Load a JSONL file into a list of objects, or [] if missing."""
        if not path.exists():
            return []
        rows = []
        with path.open() as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def append_jsonl(self, path: Path, record: dict[str, object]) -> None:
        """Append one JSON object as a line, creating parent directories."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")

    def load_generation_manifest(self) -> list[dict[str, object]]:
        """Load the generation manifest JSONL from disk."""
        return self.load_jsonl(self.generation_manifest_path())

    def load_embedding_manifest(self, subdir: str = "") -> list[dict[str, object]]:
        """Load the embedding manifest JSONL for an optional subdir."""
        return self.load_jsonl(self.embedding_manifest_path(subdir))

    def completed_generation_ids(self) -> set[str]:
        """Return record ids already present in the generation manifest."""
        return {str(row["record_id"]) for row in self.load_generation_manifest()}

    def completed_embedding_ids(self, subdir: str = "") -> set[str]:
        """Return record ids already present in the embedding manifest."""
        return {str(row["record_id"]) for row in self.load_embedding_manifest(subdir)}

    def write_run_state(self, **fields: object) -> dict[str, object]:
        """Merge fields into run_state.json, upload it, and return the state."""
        path = self.run_state_path()
        state: dict[str, object] = {}
        if path.exists():
            state = json.loads(path.read_text())
        state.update(fields)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state, indent=2))
        self.upload_run_state()
        return state


def get_storage(settings: LLMSettings | None = None) -> ArtifactStorage:
    """Construct an ArtifactStorage from settings or the environment."""
    return ArtifactStorage(settings=settings)
