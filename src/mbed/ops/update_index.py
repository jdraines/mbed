import logging
from pathlib import Path

from ..index_manager import IndexManager
from ..file_tracking import detect_changes


logger = logging.getLogger(__name__)


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

