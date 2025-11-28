"""Shared fixtures and helpers for mbed tests."""

import os
import subprocess
import sys
from pathlib import Path
from typing import List

import pytest
from llama_index.core import Settings
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.llms import MockLLM

from mbed.ops import create_index


class FastMockEmbedding(BaseEmbedding):
    """
    Fast mock embedding that generates deterministic vectors based on text hash.
    Does NOT preserve semantic similarity - only for testing plumbing.
    """

    _dimension: int = 384

    def _get_query_embedding(self, query: str) -> List[float]:
        # Generate deterministic but non-semantic embedding from hash
        hash_val = hash(query) % (2**31)
        # Create a simple vector from hash bits
        return [float((hash_val >> (i % 31)) & 1) for i in range(self._dimension)]

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._get_query_embedding(text)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)


# Configure mock LLM and embeddings for all tests (unless using real embeddings)
@pytest.fixture(scope="session", autouse=True)
def configure_test_llm():
    """Configure a mock LLM for testing to avoid needing a real LLM."""
    Settings.llm = MockLLM()

    # Use mock embeddings unless MBED_USE_REAL_EMBEDDINGS is set
    if not os.getenv("MBED_USE_REAL_EMBEDDINGS"):
        Settings.embed_model = FastMockEmbedding()


@pytest.fixture
def tmp_test_dir(tmp_path):
    """Create an isolated temporary directory for tests."""
    return tmp_path


@pytest.fixture
def create_test_documents():
    """
    Factory fixture to create test documents in a directory.

    Returns:
        Function that creates documents and returns list of paths
    """

    def _create(directory: Path, documents: dict[str, str]) -> list[Path]:
        created = []
        for filename, content in documents.items():
            file_path = directory / filename
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content)
            created.append(file_path)
        return created

    return _create


@pytest.fixture
def create_indexed_directory(create_test_documents):
    """
    Factory fixture to create a directory with documents and index it.
    Uses SimpleVectorStore (in-memory dictionary) for maximum speed.

    Returns:
        Function that creates and indexes a directory
    """

    def _create(
        directory: Path,
        documents: dict[str, str],
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        storage_type: str = "simple",
    ) -> Path:

        # Create documents
        create_test_documents(directory, documents)

        # Index the directory with simple in-memory storage for speed
        create_index(directory, model_name=model_name, storage_type=storage_type)

        return directory

    return _create


@pytest.fixture
def run_cli_command():
    """
    Factory fixture to run mbed CLI commands in-process (fast).

    Returns:
        Function that runs CLI commands and returns results
    """
    from click.testing import CliRunner
    from mbed.cli import cli

    runner = CliRunner()

    def _run(args: list[str], input_text: str = None) -> dict:
        result = runner.invoke(cli, args, input=input_text)

        return {
            "returncode": result.exit_code,
            "stdout": result.output,
            "stderr": "",  # Click captures everything in output
        }

    return _run


@pytest.fixture
def run_cli_command_subprocess():
    """
    Factory fixture to run mbed CLI commands via subprocess (slow but realistic).
    Use this for the real embeddings integration test.

    Returns:
        Function that runs CLI commands and returns results
    """

    def _run(args: list[str], input_text: str = None) -> dict:
        cmd = ["uv", "run", "mbed"] + args

        # Pass environment variables to subprocess
        env = os.environ.copy()

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            input=input_text,
            env=env,
        )

        return {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

    return _run
