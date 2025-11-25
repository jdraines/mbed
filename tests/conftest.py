"""Shared fixtures and helpers for mbed tests."""

import subprocess
import sys
from pathlib import Path

import pytest
from llama_index.core import Settings
from llama_index.core.llms import MockLLM


# Configure mock LLM for all tests
@pytest.fixture(scope="session", autouse=True)
def configure_test_llm():
    """Configure a mock LLM for testing to avoid needing a real LLM."""
    Settings.llm = MockLLM()


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

    Returns:
        Function that creates and indexes a directory
    """

    def _create(
        directory: Path,
        documents: dict[str, str],
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> Path:
        from mbed.indexer import create_index

        # Create documents
        create_test_documents(directory, documents)

        # Index the directory
        create_index(directory, model_name=model_name)

        return directory

    return _create


@pytest.fixture
def run_cli_command():
    """
    Factory fixture to run mbed CLI commands.

    Returns:
        Function that runs CLI commands and returns results
    """

    def _run(args: list[str], input_text: str = None) -> dict:
        cmd = ["uv", "run", "mbed"] + args

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            input=input_text,
        )

        return {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

    return _run
