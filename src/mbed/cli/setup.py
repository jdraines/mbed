"""CLI for mbed using Click."""

from functools import wraps
import logging
import sys

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


def setup(func):
    @wraps(func)
    def setup_and_execute(*args, **kwargs):
        v = "verbose" in kwargs
        setup_logging(v)
        setup_llm()
        return func(*args, **kwargs)

    return setup_and_execute
