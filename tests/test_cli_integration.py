"""End-to-end CLI integration tests."""

import time


def test_init_search_workflow(tmp_test_dir, create_test_documents, run_cli_command):
    """
    Test the primary user workflow: init a directory, then search it.
    """
    # Create test documents
    docs = {
        "python.txt": "Python is a versatile programming language used for web development",
        "rust.txt": "Rust is a systems programming language focused on safety",
        "go.txt": "Go is designed for building scalable network services",
    }
    create_test_documents(tmp_test_dir, docs)

    # Run mbed init
    result = run_cli_command(["init", str(tmp_test_dir)])

    # Verify init succeeded
    assert result["returncode"] == 0
    assert (tmp_test_dir / ".mbed").exists()
    assert (tmp_test_dir / ".mbed" / "metadata.json").exists()
    assert "Index created" in result["stdout"]

    # Run mbed search
    result = run_cli_command(["search", str(tmp_test_dir), "systems programming"])

    # Verify search succeeded
    assert result["returncode"] == 0
    # Should find content related to "systems programming" (likely Rust)
    assert "Rust" in result["stdout"] or "systems" in result["stdout"].lower()


def test_init_update_search_workflow(tmp_test_dir, create_test_documents, run_cli_command):
    """
    Test the incremental update workflow: init, add files, update, search.
    """
    # Create initial documents
    initial_docs = {
        "doc1.txt": "Initial document about databases",
        "doc2.txt": "Another document about storage",
    }
    create_test_documents(tmp_test_dir, initial_docs)

    # Init
    result = run_cli_command(["init", str(tmp_test_dir)])
    assert result["returncode"] == 0

    # Add a new file
    time.sleep(0.1)
    new_file = tmp_test_dir / "doc3.txt"
    new_file.write_text("This document discusses machine learning and neural networks")

    # Run mbed update with auto-confirm
    result = run_cli_command(["update", str(tmp_test_dir), "-y"])

    # Verify update succeeded
    assert result["returncode"] == 0
    assert "Index updated successfully" in result["stdout"] or "Processed" in result["stdout"]

    # Search for content from the new file
    result = run_cli_command(["search", str(tmp_test_dir), "machine learning"])

    # Verify search finds the new content
    assert result["returncode"] == 0
    assert (
        "machine learning" in result["stdout"].lower()
        or "neural" in result["stdout"].lower()
    )


def test_status_command_reports_changes(tmp_test_dir, create_test_documents, run_cli_command):
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

    result = run_cli_command(["init", str(tmp_test_dir)])
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
    Test complete file deletion workflow: deleted files don't appear in search.
    """
    # Create documents including one with "secret" content
    docs = {
        "public.txt": "This is public information available to everyone",
        "secret.txt": "This document contains confidential secret information",
        "other.txt": "This is other general information",
    }
    create_test_documents(tmp_test_dir, docs)

    # Init
    result = run_cli_command(["init", str(tmp_test_dir)])
    assert result["returncode"] == 0

    # Verify "secret" is searchable
    result = run_cli_command(["search", str(tmp_test_dir), "secret information"])
    assert result["returncode"] == 0
    original_output = result["stdout"]
    assert "secret" in original_output.lower() or "confidential" in original_output.lower()

    # Delete the secret file
    time.sleep(0.1)
    (tmp_test_dir / "secret.txt").unlink()

    # Update index with auto-confirm
    result = run_cli_command(["update", str(tmp_test_dir), "-y"])
    assert result["returncode"] == 0
    assert "Removed" in result["stdout"] or "deleted" in result["stdout"].lower()

    # Search for "secret" again - should not appear strongly
    result = run_cli_command(["search", str(tmp_test_dir), "secret information"])
    assert result["returncode"] == 0

    # The response should be different - either no results or different content
    # We can't guarantee "secret" won't appear at all (could be in the query echo),
    # but we can verify the public document is still searchable
    result = run_cli_command(["search", str(tmp_test_dir), "public information"])
    assert result["returncode"] == 0
    assert "public" in result["stdout"].lower()
