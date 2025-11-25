"""CLI for mbed using Click."""

import logging
import sys

import click
from llama_index.core import Settings
from llama_index.core.llms import MockLLM


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the application."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def setup_llm() -> None:
    """Configure a mock LLM for embeddings-only use."""
    # Always use MockLLM since we're only doing vector search, not LLM synthesis
    Settings.llm = MockLLM()


@click.group()
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose output")
@click.pass_context
def cli(ctx, verbose):
    """mbed - Minimal Embeddings with Vector Search"""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    setup_logging(verbose)
    setup_llm()


# Import commands to register them with the group
from . import commands  # noqa: E402, F401
