"""End-to-end CLI integration tests."""

import json
import os
import time

import pytest
from mbed.metadata import Metadata


def test_init_search_workflow(tmp_test_dir, create_test_documents, run_cli_command):
    """
    Test the primary user workflow: init a directory, then search it.
    Tests plumbing only - does not verify semantic search quality.
    """
    # Create test documents
    docs = {
        "python.txt": "Python is a versatile programming language used for web development",
        "rust.txt": "Rust is a systems programming language focused on safety",
        "go.txt": "Go is designed for building scalable network services",
        "secret.env": "SECRET_KEY=my_secret_value",
    }
    create_test_documents(tmp_test_dir, docs)

    # Run mbed init with exclude pattern for .env files
    result = run_cli_command(["init", "-d", str(tmp_test_dir), "--exclude", "*.env"])

    # Verify init succeeded
    assert result["returncode"] == 0
    assert (tmp_test_dir / ".mbed").exists()
    assert (tmp_test_dir / ".mbed" / "metadata.json").exists()
    assert "Index created" in result["stdout"]

    # Verify the .env file was excluded from indexing
    metadata_path = tmp_test_dir / ".mbed" / "metadata.json"
    metadata = Metadata(**json.loads(metadata_path.read_text()))
    indexed_files = metadata.indexed_files
    assert "secret.env" not in indexed_files
    assert "python.txt" in indexed_files
    assert "rust.txt" in indexed_files
    assert "go.txt" in indexed_files

    # Run mbed search
    result = run_cli_command(["search", "systems programming", "-d", str(tmp_test_dir)])

    # Verify search succeeded (returns results, not checking quality)
    assert result["returncode"] == 0
    assert len(result["stdout"]) > 0  # Got some output


def test_init_update_search_workflow(
    tmp_test_dir, create_test_documents, run_cli_command
):
    """
    Test the incremental update workflow: init, add files, update, search.
    Tests plumbing only - does not verify semantic search quality.
    """
    # Create initial documents
    initial_docs = {
        "doc1.txt": "Initial document about databases",
        "doc2.txt": "Another document about storage",
    }
    create_test_documents(tmp_test_dir, initial_docs)

    # Init
    result = run_cli_command(["init", "-d", str(tmp_test_dir)])
    assert result["returncode"] == 0

    # Add a new file
    time.sleep(0.1)
    new_file = tmp_test_dir / "doc3.txt"
    new_file.write_text("This document discusses machine learning and neural networks")

    # Run mbed update with auto-confirm
    result = run_cli_command(["update", "-d", str(tmp_test_dir), "-y"])

    # Verify update succeeded
    assert result["returncode"] == 0
    assert (
        "Index updated successfully" in result["stdout"]
        or "Processed" in result["stdout"]
    )

    # Search for content from the new file (just verify search works)
    result = run_cli_command(["search", "machine learning", "-d", str(tmp_test_dir)])

    # Verify search returns results
    assert result["returncode"] == 0
    assert len(result["stdout"]) > 0  # Got some output


def test_status_command_reports_changes(
    tmp_test_dir, create_test_documents, run_cli_command
):
    """
    Test that the status command correctly reports file changes.
    """
    # Create and index initial documents
    initial_docs = {
        "fileA.txt": "Content A",
        "fileB.txt": "Content B",
        "fileC.txt": "Content C",
    }
    create_test_documents(tmp_test_dir, initial_docs)

    result = run_cli_command(["init", "-d", str(tmp_test_dir)])
    assert result["returncode"] == 0

    # Make changes
    time.sleep(0.1)

    # Add new file
    (tmp_test_dir / "fileD.txt").write_text("New content")

    # Modify existing file
    (tmp_test_dir / "fileB.txt").write_text("Modified content - much longer")

    # Delete existing file
    (tmp_test_dir / "fileC.txt").unlink()

    # Run status command
    result = run_cli_command(["status", str(tmp_test_dir)])

    # Verify status output
    assert result["returncode"] == 0
    assert "Added:" in result["stdout"] or "added" in result["stdout"].lower()
    assert "Modified:" in result["stdout"] or "modified" in result["stdout"].lower()
    assert "Deleted:" in result["stdout"] or "deleted" in result["stdout"].lower()

    # Verify counts (1 added, 1 modified, 1 deleted)
    assert "1" in result["stdout"]  # Should show counts of 1


def test_file_deletion_end_to_end(tmp_test_dir, create_test_documents, run_cli_command):
    """
    Test complete file deletion workflow: init, delete file, update.
    Tests plumbing only - does not verify semantic search quality.
    """
    # Create documents including one with "secret" content
    docs = {
        "public.txt": "This is public information available to everyone",
        "secret.txt": "This document contains confidential secret information",
        "other.txt": "This is other general information",
    }
    create_test_documents(tmp_test_dir, docs)

    # Init
    result = run_cli_command(["init", "-d", str(tmp_test_dir)])
    assert result["returncode"] == 0

    # Verify search works
    result = run_cli_command(["search", "secret information", "-d", str(tmp_test_dir)])
    assert result["returncode"] == 0

    # Delete the secret file
    time.sleep(0.1)
    (tmp_test_dir / "secret.txt").unlink()

    # Update index with auto-confirm
    result = run_cli_command(["update", "-d", str(tmp_test_dir), "-y"])
    assert result["returncode"] == 0
    assert "Removed" in result["stdout"] or "deleted" in result["stdout"].lower()

    # Verify search still works after deletion
    result = run_cli_command(["search", "public information", "-d", str(tmp_test_dir)])
    assert result["returncode"] == 0
    assert len(result["stdout"]) > 0  # Got some output


@pytest.mark.skipif(
    not os.getenv("MBED_USE_REAL_EMBEDDINGS"),
    reason="Real embedding test only runs when MBED_USE_REAL_EMBEDDINGS is set",
)
def test_semantic_search_with_real_embeddings(
    tmp_test_dir, create_test_documents, run_cli_command_subprocess
):
    """
    Integration test with real embeddings to verify semantic search quality.
    Only runs when MBED_USE_REAL_EMBEDDINGS=1 is set.

    This test is slower but verifies that semantic search actually works.
    Uses subprocess to ensure realistic end-to-end testing.
    """
    # Create test documents with clear semantic distinctions
    docs = {
        "rust.txt": "Rust is a systems programming language that focuses on safety and performance",
        "python.txt": "Python is a high-level interpreted language great for scripting",
        "cooking.txt": "Baking bread requires flour, water, yeast, and patience",
    }
    create_test_documents(tmp_test_dir, docs)

    # Init
    result = run_cli_command_subprocess(["init", "-d", str(tmp_test_dir)])
    assert result["returncode"] == 0

    # Search for systems programming - should find Rust
    result = run_cli_command_subprocess(
        ["search", "systems programming", "-d", str(tmp_test_dir), "--top-k", "1"]
    )
    assert result["returncode"] == 0
    assert (
        "rust.txt" in result["stdout"].lower() or "safety" in result["stdout"].lower()
    )

    # Search for baking - should find cooking document
    result = run_cli_command_subprocess(
        ["search", "baking recipes", "-d", str(tmp_test_dir), "--top-k", "1"]
    )
    assert result["returncode"] == 0
    assert "cooking.txt" in result["stdout"].lower()
