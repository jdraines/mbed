import logging
from dataclasses import dataclass
from pathlib import Path

from .index_manager import IndexManager
from .metadata import MetadataManager

logger = logging.getLogger(__name__)


@dataclass
class FileChange:
    """Represents a file change."""

    path: Path
    change_type: str  # "added", "modified", "deleted"
    old_mtime: float | None = None
    new_mtime: float | None = None


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
    indexed_files = metadata["indexed_files"]

    changes: dict[str, list[FileChange]] = {
        "added": [],
        "modified": [],
        "deleted": [],
    }

    # Scan current directory
    current_files = {}
    for file_path in directory.rglob("*"):
        if file_path.is_file() and not str(file_path).startswith(str(mbed_dir)):
            rel_path = str(file_path.relative_to(directory))
            stat = file_path.stat()
            current_files[rel_path] = {
                "path": file_path,
                "mtime": stat.st_mtime,
                "size": stat.st_size,
            }

    # Check for added and modified files
    for rel_path, file_info in current_files.items():
        if rel_path not in indexed_files:
            changes["added"].append(
                FileChange(
                    path=file_info["path"],
                    change_type="added",
                    new_mtime=file_info["mtime"],
                )
            )
        else:
            old_mtime = indexed_files[rel_path]["mtime"]
            new_mtime = file_info["mtime"]

            # Check if modified (compare mtime and size)
            if (
                new_mtime != old_mtime
                or file_info["size"] != indexed_files[rel_path]["size"]
            ):
                changes["modified"].append(
                    FileChange(
                        path=file_info["path"],
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
                    path=Path(file_info["path"]),
                    change_type="deleted",
                    old_mtime=file_info["mtime"],
                )
            )

    return changes


def update_index(directory: Path) -> dict:
    """
    Update index with all detected changes.

    Args:
        directory: Root directory being indexed

    Returns:
        Dictionary with update results:
        - changes: The detected changes
        - processed: Number of files processed
        - errors: List of (file_path, error_message) tuples
    """
    changes = detect_changes(directory)

    total_changes = (
        len(changes["added"]) + len(changes["modified"]) + len(changes["deleted"])
    )

    if total_changes == 0:
        return {
            "changes": changes,
            "processed": 0,
            "errors": [],
            "no_changes": True,
        }

    logger.info(f"Updating index with {total_changes} changes")

    # Load index manager
    manager = IndexManager(directory)
    manager.load()

    # Process added and modified files
    if changes["added"] or changes["modified"]:
        files_to_add = [c.path for c in changes["added"]] + [
            c.path for c in changes["modified"]
        ]

        result = manager.add_files(files_to_add)
        processed = result["processed"]
        errors = result["errors"]
    else:
        processed = 0
        errors = []

    # Handle deleted files
    if changes["deleted"]:
        deleted_paths = [c.path for c in changes["deleted"]]
        result = manager.remove_files(deleted_paths)
        # Note: result contains removed count and any errors
        if result["errors"]:
            logger.warning(
                f"Encountered {len(result['errors'])} errors while removing files"
            )

    # Save metadata
    manager.save_metadata()

    logger.info("Index updated successfully")

    return {
        "changes": changes,
        "processed": processed,
        "errors": errors,
        "no_changes": False,
    }
