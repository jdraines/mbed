"""Click commands for mbed CLI."""

from pathlib import Path
from typing import Annotated

import typer
from rich import print

from ...file_tracking import detect_changes
from ..setup import setup


status = typer.Typer()


@status.command("status")
@setup
def status_(directory: Annotated[Path, typer.Argument()] = Path(".")):
    """Check for file changes."""

    directory = Path(directory).resolve()
    changes = detect_changes(directory)
    total = len(changes["added"]) + len(changes["modified"]) + len(changes["deleted"])

    if total == 0:
        print("No changes detected. Index is up to date.")
    else:
        print("File changes detected:")
        print(f"  Added:    {len(changes['added'])} files")
        print(f"  Modified: {len(changes['modified'])} files")
        print(f"  Deleted:  {len(changes['deleted'])} files")
        print("\nRun 'mbed update' to apply changes.")
