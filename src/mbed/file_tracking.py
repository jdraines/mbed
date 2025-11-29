import logging
from dataclasses import dataclass
from pathlib import Path

from .metadata import FileMetadata, MetadataManager


logger = logging.getLogger(__name__)


@dataclass
class FileChange:
    """Represents a file change."""

    path: Path
    change_type: str  # "added", "modified", "deleted"
    old_mtime: float | None = None
    new_mtime: float | None = None


def _should_exclude(file_path: Path, directory: Path, exclude_patterns: list[str]) -> bool:
    """
    Check if a file should be excluded based on patterns.

    Args:
        file_path: Absolute path to the file
        directory: Root directory being indexed
        exclude_patterns: List of glob patterns to exclude

    Returns:
        True if file should be excluded, False otherwise
    """
    if not exclude_patterns:
        return False

    rel_path = file_path.relative_to(directory)

    for pattern in exclude_patterns:
        # Check if the pattern matches the relative path
        if rel_path.match(pattern):
            return True
        # Also check if any parent directory matches
        if any(part == pattern or Path(part).match(pattern) for part in rel_path.parts):
            return True

    return False


def detect_changes(directory: Path) -> dict[str, list[FileChange]]:
    """
    Detect file changes in indexed directory.

    Args:
        directory: Root directory being indexed

    Returns:
        Dictionary with keys: "added", "modified", "deleted"
    """
    mbed_dir = directory / ".mbed"

    if not mbed_dir.exists():
        raise ValueError(f"Directory {directory} is not indexed.")

    # Load metadata
    metadata_mgr = MetadataManager(mbed_dir)
    metadata = metadata_mgr.load_metadata()
    indexed_files = metadata.indexed_files
    config = metadata.config
    exclude_patterns = config.exclude

    changes: dict[str, list[FileChange]] = {
        "added": [],
        "modified": [],
        "deleted": [],
    }

    # Scan current directory
    current_files = {}
    for file_path in directory.rglob("*"):
        if file_path.is_file() and not str(file_path).startswith(str(mbed_dir)):
            # Skip excluded files
            if _should_exclude(file_path, directory, exclude_patterns):
                continue

            rel_path = str(file_path.relative_to(directory))
            stat = file_path.stat()
            current_files[rel_path] = FileMetadata(
                path=str(file_path),
                mtime=stat.st_mtime,
                size=stat.st_size,
            )
    # Check for added and modified files
    for rel_path, file_info in current_files.items():
        if rel_path not in indexed_files:
            changes["added"].append(
                FileChange(
                    path=file_info.path,
                    change_type="added",
                    new_mtime=file_info.mtime,
                )
            )
        else:
            old_mtime = indexed_files[rel_path].mtime
            new_mtime = file_info.mtime

            # Check if modified (compare mtime and size)
            if (
                new_mtime != old_mtime
                or file_info.size != indexed_files[rel_path].size
            ):
                changes["modified"].append(
                    FileChange(
                        path=file_info.path,
                        change_type="modified",
                        old_mtime=old_mtime,
                        new_mtime=new_mtime,
                    )
                )

    # Check for deleted files
    for rel_path, file_info in indexed_files.items():
        if rel_path not in current_files:
            changes["deleted"].append(
                FileChange(
                    path=Path(file_info.path),
                    change_type="deleted",
                    old_mtime=file_info.mtime,
                )
            )

    return changes