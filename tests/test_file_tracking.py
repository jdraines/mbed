"""Tests for file change detection logic."""

import time

from mbed.file_tracking import detect_changes


def test_detect_new_files(tmp_test_dir, create_indexed_directory):
    """
    Test that newly added files are detected correctly.
    """
    # Create initial indexed directory with two files
    initial_docs = {
        "fileA.txt": "Content of file A",
        "fileB.txt": "Content of file B",
    }
    create_indexed_directory(tmp_test_dir, initial_docs)

    # Add a new file
    new_file = tmp_test_dir / "fileC.txt"
    new_file.write_text("Content of file C")

    # Detect changes
    changes = detect_changes(tmp_test_dir)

    # Verify fileC is detected as added
    assert len(changes["added"]) == 1
    assert changes["added"][0].path == new_file
    assert changes["added"][0].change_type == "added"

    # No modifications or deletions
    assert len(changes["modified"]) == 0
    assert len(changes["deleted"]) == 0


def test_detect_modified_files(tmp_test_dir, create_indexed_directory):
    """
    Test that modified files are detected correctly.
    """
    # Create initial indexed directory
    initial_docs = {
        "fileA.txt": "Original content",
        "fileB.txt": "Unchanged content",
    }
    create_indexed_directory(tmp_test_dir, initial_docs)

    # Wait a moment to ensure mtime changes
    time.sleep(0.1)

    # Modify fileA
    file_to_modify = tmp_test_dir / "fileA.txt"
    file_to_modify.write_text("Modified content - now much longer")

    # Detect changes
    changes = detect_changes(tmp_test_dir)

    # Verify fileA is detected as modified
    assert len(changes["modified"]) == 1
    assert changes["modified"][0].path == file_to_modify
    assert changes["modified"][0].change_type == "modified"
    assert changes["modified"][0].old_mtime is not None
    assert changes["modified"][0].new_mtime is not None

    # No additions or deletions
    assert len(changes["added"]) == 0
    assert len(changes["deleted"]) == 0


def test_detect_deleted_files(tmp_test_dir, create_indexed_directory):
    """
    Test that deleted files are detected correctly.
    """
    # Create initial indexed directory
    initial_docs = {
        "fileA.txt": "Content of file A",
        "fileB.txt": "Content of file B",
        "fileC.txt": "Content of file C",
    }
    create_indexed_directory(tmp_test_dir, initial_docs)

    # Delete fileB
    file_to_delete = tmp_test_dir / "fileB.txt"
    file_to_delete.unlink()

    # Detect changes
    changes = detect_changes(tmp_test_dir)

    # Verify fileB is detected as deleted
    assert len(changes["deleted"]) == 1
    assert changes["deleted"][0].path == file_to_delete
    assert changes["deleted"][0].change_type == "deleted"
    assert changes["deleted"][0].old_mtime is not None

    # No additions or modifications
    assert len(changes["added"]) == 0
    assert len(changes["modified"]) == 0


def test_detect_multiple_changes_simultaneously(tmp_test_dir, create_indexed_directory):
    """
    Test that multiple types of changes are detected correctly in one scan.
    """
    # Create initial indexed directory
    initial_docs = {
        "fileA.txt": "Content A",
        "fileB.txt": "Content B",
        "fileC.txt": "Content C",
    }
    create_indexed_directory(tmp_test_dir, initial_docs)

    # Wait to ensure mtime changes
    time.sleep(0.1)

    # Add a new file
    new_file = tmp_test_dir / "fileD.txt"
    new_file.write_text("New content D")

    # Modify an existing file
    modify_file = tmp_test_dir / "fileB.txt"
    modify_file.write_text("Modified content B - much longer now")

    # Delete an existing file
    delete_file = tmp_test_dir / "fileC.txt"
    delete_file.unlink()

    # Detect changes
    changes = detect_changes(tmp_test_dir)

    # Verify all three types of changes detected
    assert len(changes["added"]) == 1
    assert changes["added"][0].path == new_file

    assert len(changes["modified"]) == 1
    assert changes["modified"][0].path == modify_file

    assert len(changes["deleted"]) == 1
    assert changes["deleted"][0].path == delete_file

    # Verify fileA remains unchanged (not in any list)
    all_changed_files = (
        [c.path for c in changes["added"]]
        + [c.path for c in changes["modified"]]
        + [c.path for c in changes["deleted"]]
    )
    assert (tmp_test_dir / "fileA.txt") not in all_changed_files
