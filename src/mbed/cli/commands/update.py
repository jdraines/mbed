"""Click commands for mbed CLI."""

from typing import Annotated
from pathlib import Path

import typer
from rich import print

from ...ops import update_index
from ...file_tracking import detect_changes
from ..setup import setup

update = typer.Typer()


@update.command("update")
@setup
def update_(
    directory: Annotated[Path, typer.Option("--directory", "-d")] = Path("."),
    yes: Annotated[bool, typer.Option("--yes", "-y")] = False,
):
    """Update index with new/modified files."""

    directory = Path(directory).resolve()

    changes = detect_changes(directory)
    total_changes = (
        len(changes["added"]) + len(changes["modified"]) + len(changes["deleted"])
    )

    if total_changes == 0:
        print("No changes detected. Index is up to date.")
        return

    print("Detected changes:")
    if changes["added"]:
        print(f"  Added: {len(changes['added'])} files")
        for change in changes["added"][:5]:
            print(f"    + {change.path.name}")
        if len(changes["added"]) > 5:
            print(f"    ... and {len(changes['added']) - 5} more")

    if changes["modified"]:
        print(f"  Modified: {len(changes['modified'])} files")
        for change in changes["modified"][:5]:
            print(f"    ~ {change.path.name}")
        if len(changes["modified"]) > 5:
            print(f"    ... and {len(changes['modified']) - 5} more")

    if changes["deleted"]:
        print(f"  Deleted: {len(changes['deleted'])} files")
        for change in changes["deleted"][:5]:
            print(f"    - {change.path.name}")
        if len(changes["deleted"]) > 5:
            print(f"    ... and {len(changes['deleted']) - 5} more")

    if not yes:
        if not typer.confirm("\nApply changes?", abort=True):
            print("Cancelled.")
            return

    result = update_index(directory)

    if result["errors"]:
        print(
            f"\nProcessed {result['processed']} files with {len(result['errors'])} errors:"
        )
        for file_path, error in result["errors"]:
            print(f"  Error: {file_path.name} - {error}")
    else:
        print(f"\nIndex updated successfully! Processed {result['processed']} files.")

    if changes["deleted"]:
        deleted_count = len(changes["deleted"])
        print(f"\nRemoved {deleted_count} deleted file(s) from index.")
