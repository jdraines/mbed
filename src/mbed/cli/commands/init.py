"""Click commands for mbed CLI."""

from pathlib import Path
from typing import Literal, List, Annotated, TypeAlias, Optional

import typer
from rich import print

from ...ops import create_index
from ..setup import setup

MultipleStrOpt: TypeAlias = Annotated[Optional[List[str]], typer.Option()]

init = typer.Typer()


@init.command("init")
@setup
def init_(
    directory: Annotated[Path, typer.Option("--directory", "-d")] = Path("."),
    model: str = "sentence-transformers/all-MiniLM-L6-v2",
    storage: Literal["chromadb"] = "chromadb",
    top_k: int = 3,
    exclude: MultipleStrOpt = None,
):
    """Initialize index for a directory."""
    exclude = exclude or []
    directory = Path(directory).resolve()
    exclude_list = list(exclude) if exclude else None
    create_index(directory, model, storage, top_k=top_k, exclude=exclude_list)
    print(f"Index created at {directory / '.mbed'}")
