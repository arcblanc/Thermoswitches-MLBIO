import json
from pathlib import Path

from data_engineering.paths import resolve_path
from validation_embedding.config import LLMSettings, load_llm_settings

DE_NOVO_GCS_SUBDIR = "de_novo"
EMBEDDING_GCS_SUBDIR = "validation_embedding"
RUN_STATE_NAME = "run_state.json"
GENERATION_MANIFEST_NAME = "generation_manifest.jsonl"
EMBEDDING_MANIFEST_NAME = "manifest.jsonl"
GENERATED_FASTA_NAME = "generated.fasta"


class ArtifactStorage:
    def __init__(self, settings: LLMSettings | None = None):
        self.settings = settings or load_llm_settings()
        self._client = None

    @property
    def uses_gcs(self) -> bool:
        return self.settings.storage_target == "gcs"

    def _require_gcs(self):
        if not self.settings.gcs_bucket:
            raise ValueError("GCS_BUCKET is required when STORAGE_TARGET=gcs")

    def _gcs_blob_path(self, relative_path: str) -> str:
        prefix = self.settings.gcs_prefix
        relative_path = relative_path.lstrip("/")
        return f"{prefix}/{relative_path}" if prefix else relative_path

    def _client_or_raise(self):
        self._require_gcs()
        if self._client is None:
            from google.cloud import storage

            self._client = storage.Client()
        return self._client

    def _bucket(self):
        client = self._client_or_raise()
        return client.bucket(self.settings.gcs_bucket)

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

    def upload_file(self, local_path: Path, gcs_relative: str | None = None) -> None:
        if not self.uses_gcs:
            return
        local_path = Path(local_path)
        if not local_path.exists():
            return
        gcs_relative = gcs_relative or str(local_path.relative_to(resolve_path(".")))
        blob = self._bucket().blob(self._gcs_blob_path(gcs_relative))
        blob.upload_from_filename(str(local_path))

    def download_file(self, gcs_relative: str, local_path: Path) -> bool:
        if not self.uses_gcs:
            return local_path.exists()
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        blob = self._bucket().blob(self._gcs_blob_path(gcs_relative))
        if not blob.exists():
            return False
        blob.download_to_filename(str(local_path))
        return True

    def upload_de_novo_artifacts(self) -> None:
        if not self.uses_gcs:
            return
        fasta = self.de_novo_fasta_path()
        manifest = self.generation_manifest_path()
        if fasta.exists():
            self.upload_file(
                fasta,
                f"{DE_NOVO_GCS_SUBDIR}/{GENERATED_FASTA_NAME}",
            )
        if manifest.exists():
            self.upload_file(
                manifest,
                f"{DE_NOVO_GCS_SUBDIR}/{GENERATION_MANIFEST_NAME}",
            )

    def upload_embedding_artifacts(self, subdir: str = "") -> None:
        if not self.uses_gcs:
            return
        output_dir = self.embedding_dir(subdir)
        if not output_dir.exists():
            return
        prefix = EMBEDDING_GCS_SUBDIR
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
        if not self.uses_gcs:
            return
        self.download_file(
            f"{DE_NOVO_GCS_SUBDIR}/{GENERATED_FASTA_NAME}",
            self.de_novo_fasta_path(),
        )
        self.download_file(
            f"{DE_NOVO_GCS_SUBDIR}/{GENERATION_MANIFEST_NAME}",
            self.generation_manifest_path(),
        )
        prefix = EMBEDDING_GCS_SUBDIR
        if embedding_subdir:
            prefix = f"{prefix}/{embedding_subdir}"
        output_dir = self.embedding_dir(embedding_subdir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.download_file(
            f"{prefix}/{EMBEDDING_MANIFEST_NAME}",
            self.embedding_manifest_path(embedding_subdir),
        )
        bucket = self._bucket()
        blob_prefix = self._gcs_blob_path(f"{prefix}/")
        for blob in bucket.list_blobs(prefix=blob_prefix):
            name = blob.name[len(blob_prefix) :]
            if not name or "/" in name:
                continue
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
