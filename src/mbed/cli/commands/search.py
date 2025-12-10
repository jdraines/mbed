"""Click commands for mbed CLI."""

from pathlib import Path
from typing import Optional, Annotated

import typer
from rich import print

from ...ops import search_directory
from ..setup import setup

search = typer.Typer()


@search.command("search")
@setup
def search_(
    query: str,
    directory: Annotated[Path, typer.Option("--directory", "-d")] = Path("."),
    top_k: Optional[int] = None,
):
    """Search indexed directory."""
    directory = Path(directory).resolve()
    response = search_directory(directory, query, top_k)
    print(response)
