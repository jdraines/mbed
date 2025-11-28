"""Click commands for mbed CLI."""

from pathlib import Path

import click

from ..ops import (
    create_index,
    search_directory,
    update_index
)
from ..file_tracking import detect_changes


@click.command()
@click.option("--directory", "-d", type=click.Path(path_type=Path), default=".")
@click.option(
    "--model",
    default="sentence-transformers/all-MiniLM-L6-v2",
    help="HuggingFace embedding model to use",
)
@click.option(
    "--storage",
    type=click.Choice(["chromadb"]),
    default="chromadb",
    help="Vector storage backend",
)
@click.option("--top-k", type=int, default=3, help="Default number of search results")
@click.option(
    "--exclude",
    multiple=True,
    help="File patterns to exclude (can be used multiple times)",
)
def init(directory, model, storage, top_k, exclude):
    """Initialize index for a directory."""
    directory = Path(directory).resolve()
    exclude_list = list(exclude) if exclude else None
    create_index(directory, model, storage, top_k=top_k, exclude=exclude_list)
    click.echo(f"Index created at {directory / '.mbed'}")


@click.command()
@click.argument("query")
@click.option("--directory", "-d", type=click.Path(path_type=Path), default=".")
@click.option("--top-k", type=int, default=None, help="Override number of results")
def search(directory, query, top_k):
    """Search indexed directory."""
    directory = Path(directory).resolve()
    response = search_directory(directory, query, top_k)
    click.echo(response)


@click.command()
@click.option("--directory", "-d", type=click.Path(path_type=Path), default=".")
@click.option("-y", "--yes", is_flag=True, help="Auto-confirm changes")
def update(directory, yes):
    """Update index with new/modified files."""

    directory = Path(directory).resolve()

    # First detect changes
    changes = detect_changes(directory)
    total_changes = (
        len(changes["added"]) + len(changes["modified"]) + len(changes["deleted"])
    )

    if total_changes == 0:
        click.echo("No changes detected. Index is up to date.")
        return

    # Display changes
    click.echo("Detected changes:")
    if changes["added"]:
        click.echo(f"  Added: {len(changes['added'])} files")
        for change in changes["added"][:5]:
            click.echo(f"    + {change.path.name}")
        if len(changes["added"]) > 5:
            click.echo(f"    ... and {len(changes['added']) - 5} more")

    if changes["modified"]:
        click.echo(f"  Modified: {len(changes['modified'])} files")
        for change in changes["modified"][:5]:
            click.echo(f"    ~ {change.path.name}")
        if len(changes["modified"]) > 5:
            click.echo(f"    ... and {len(changes['modified']) - 5} more")

    if changes["deleted"]:
        click.echo(f"  Deleted: {len(changes['deleted'])} files")
        for change in changes["deleted"][:5]:
            click.echo(f"    - {change.path.name}")
        if len(changes["deleted"]) > 5:
            click.echo(f"    ... and {len(changes['deleted']) - 5} more")

    # Confirm if not auto-confirmed
    if not yes:
        if not click.confirm("\nApply changes?"):
            click.echo("Cancelled.")
            return

    # Apply updates
    result = update_index(directory)

    # Display results
    if result["errors"]:
        click.echo(
            f"\nProcessed {result['processed']} files with {len(result['errors'])} errors:"
        )
        for file_path, error in result["errors"]:
            click.echo(f"  Error: {file_path.name} - {error}")
    else:
        click.echo(
            f"\nIndex updated successfully! Processed {result['processed']} files."
        )

    if changes["deleted"]:
        deleted_count = len(changes["deleted"])
        click.echo(f"\nRemoved {deleted_count} deleted file(s) from index.")



@click.command()
@click.argument("directory", type=click.Path(path_type=Path), default=".")
def status(directory):
    """Check for file changes."""

    directory = Path(directory).resolve()
    changes = detect_changes(directory)
    total = (
        len(changes["added"]) + len(changes["modified"]) + len(changes["deleted"])
    )

    if total == 0:
        click.echo("No changes detected. Index is up to date.")
    else:
        click.echo("File changes detected:")
        click.echo(f"  Added:    {len(changes['added'])} files")
        click.echo(f"  Modified: {len(changes['modified'])} files")
        click.echo(f"  Deleted:  {len(changes['deleted'])} files")
        click.echo("\nRun 'mbed update' to apply changes.")
