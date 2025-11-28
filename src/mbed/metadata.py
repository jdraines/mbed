import json
from typing import Literal
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict


class FileMetadata(BaseModel):
    path: str
    mtime: float
    size: int
    doc_ids: list[str]
    indexed_at: str


class MetadataConfig(BaseModel):
    top_k: int
    exclude: list[str]


class Metadata(BaseModel):
    model_config = ConfigDict(protected_namespaces=[])
    model_name: str
    storage_type: Literal["chromadb", "simple"]
    created_at: str
    last_updated: str
    indexed_files: dict[str, FileMetadata]
    config: MetadataConfig


class MetadataManager:
    """Manages metadata for indexed directories."""

    def __init__(self, mbed_dir: Path):
        self.mbed_dir = mbed_dir
        self.metadata_file = mbed_dir / "metadata.json"

    def save_metadata(self, metadata: dict[str, Any]) -> None:
        """Save metadata to JSON file."""
        with open(self.metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

    def load_metadata(self) -> dict[str, Any]:
        """Load metadata from JSON file."""
        if not self.metadata_file.exists():
            raise FileNotFoundError(
                f"Metadata file not found: {self.metadata_file}"
            )

        with open(self.metadata_file, "r") as f:
            metadata = json.load(f)

        Metadata.model_validate(metadata)  # type: ignore
        return metadata

    def metadata_exists(self) -> bool:
        """Check if metadata file exists."""
        return self.metadata_file.exists()
