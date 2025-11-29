import json
from typing import Literal
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, field_serializer, model_validator, Field


class FileMetadata(BaseModel):
    path: Path
    mtime: float
    size: int
    doc_ids: list[str] = Field(default_factory=list)
    indexed_at: datetime | None = None

    @model_validator(mode="before")
    def validate_indexed_at(cls, values: dict[str, Any]) -> dict[str, Any]:
        if "indexed_at" in values and isinstance(values["indexed_at"], str):
            values["indexed_at"] = datetime.fromisoformat(
                values["indexed_at"].replace("Z", "+00:00")
            )
        return values

    @model_validator(mode="before")
    def validate_path(cls, values: dict[str, Any]) -> dict[str, Any]:
        if "path" in values and isinstance(values["path"], str):
            values["path"] = Path(values["path"])
        return values

    @field_serializer("indexed_at")
    def serialize_last_updated(self, dt: datetime) -> str:
        return self._serialize_datetime(dt)
    
    def _serialize_datetime(self, dt: datetime) -> str:
        return dt.isoformat().replace("+00:00", "Z")


class MetadataConfig(BaseModel):
    top_k: int =  3
    exclude: list[str] = Field(default_factory=list)


class Metadata(BaseModel):
    model_config = ConfigDict(protected_namespaces=[])
    model_name: str
    storage_type: Literal["chromadb", "simple"]
    created_at: datetime
    last_updated: datetime
    indexed_files: dict[str, FileMetadata] = Field(default_factory=dict)
    config: MetadataConfig = Field(default_factory=MetadataConfig)

    @model_validator(mode="before")
    def validate_datetimes(cls, values: dict[str, Any]) -> dict[str, Any]:
        for field in ["created_at", "last_updated"]:
            if field in values and isinstance(values[field], str):
                values[field] = datetime.fromisoformat(
                    values[field].replace("Z", "+00:00")
                )
        return values

    @field_serializer("created_at")
    def serialize_created_at(self, dt: datetime) -> str:
        return self._serialize_datetime(dt)
    
    @field_serializer("last_updated")
    def serialize_last_updated(self, dt: datetime) -> str:
        return self._serialize_datetime(dt)
    
    def _serialize_datetime(self, dt: datetime) -> str:
        return dt.isoformat().replace("+00:00", "Z")


class MetadataManager:
    """Manages metadata for indexed directories."""

    def __init__(self, mbed_dir: Path):
        self.mbed_dir = mbed_dir
        self.metadata_file = mbed_dir / "metadata.json"

    def save_metadata(self, metadata: Metadata) -> None:
        """Save metadata to JSON file."""
        with open(self.metadata_file, "w") as f:
            f.write(metadata.model_dump_json(indent=4))

    def load_metadata(self) -> Metadata:
        """Load metadata from JSON file."""
        if not self.metadata_file.exists():
            raise FileNotFoundError(
                f"Metadata file not found: {self.metadata_file}"
            )

        with open(self.metadata_file, "r") as f:
            metadata = json.load(f)

        return Metadata(**metadata)  # type: ignore

    def metadata_exists(self) -> bool:
        """Check if metadata file exists."""
        return self.metadata_file.exists()
