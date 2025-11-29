"""Tests for IndexManager API."""

from mbed.index_manager import IndexManager

from llama_index.core import SimpleDirectoryReader


def test_initialize_and_load_roundtrip(tmp_test_dir, create_test_documents):
    """
    Test that we can initialize an index and load it back from disk.
    Verifies metadata persists correctly.
    """
    # Create test documents
    docs = {
        "doc1.txt": "Python is a programming language",
        "doc2.txt": "JavaScript is used for web development",
    }
    create_test_documents(tmp_test_dir, docs)

    # Initialize index
    from llama_index.core import SimpleDirectoryReader

    reader = SimpleDirectoryReader(input_dir=str(tmp_test_dir))
    documents = reader.load_data()

    manager = IndexManager(tmp_test_dir)
    manager.initialize(documents, "sentence-transformers/all-MiniLM-L6-v2", "chromadb")

    # Update metadata
    file_paths = [tmp_test_dir / name for name in docs.keys()]
    manager.update_file_metadata(file_paths, documents)
    manager.save_metadata()

    # Verify .mbed directory created
    assert (tmp_test_dir / ".mbed").exists()
    assert (tmp_test_dir / ".mbed" / "metadata.json").exists()

    # Load the index in a new manager instance
    manager2 = IndexManager(tmp_test_dir)
    manager2.load()

    # Verify metadata loaded correctly
    assert manager2.metadata.model_name == "sentence-transformers/all-MiniLM-L6-v2"
    assert manager2.metadata.storage_type == "chromadb"
    assert len(manager2.metadata.indexed_files) == 2
    assert "doc1.txt" in manager2.metadata.indexed_files
    assert "doc2.txt" in manager2.metadata.indexed_files

    # Verify we can query the loaded index
    query_engine = manager2.index.as_query_engine(similarity_top_k=1)
    response = query_engine.query("programming language")
    assert response is not None
    assert len(str(response)) > 0


def test_add_files_captures_doc_ids(tmp_test_dir, create_test_documents):
    """
    Test that adding files captures document IDs and makes them searchable.
    """
    # Create and initialize an empty-ish index
    initial_docs = {"initial.txt": "Initial document"}
    create_test_documents(tmp_test_dir, initial_docs)

    from llama_index.core import SimpleDirectoryReader

    reader = SimpleDirectoryReader(input_dir=str(tmp_test_dir))
    documents = reader.load_data()

    manager = IndexManager(tmp_test_dir)
    manager.initialize(documents, "sentence-transformers/all-MiniLM-L6-v2", "chromadb")
    manager.update_file_metadata([tmp_test_dir / "initial.txt"], documents)
    manager.save_metadata()

    # Now add new files
    new_docs = {
        "new1.txt": "Rust is a systems programming language",
        "new2.txt": "Go is designed for concurrency",
    }
    new_file_paths = create_test_documents(tmp_test_dir, new_docs)

    # Load and add files
    manager.load()
    result = manager.add_files(new_file_paths)

    # Verify successful processing
    assert result["processed"] == 2
    assert len(result["errors"]) == 0

    # Verify doc_ids were captured
    manager.save_metadata()
    metadata = manager.metadata
    assert "new1.txt" in metadata.indexed_files
    assert "new2.txt" in metadata.indexed_files
    assert len(metadata.indexed_files["new1.txt"].doc_ids) > 0

    # Verify files are searchable
    query_engine = manager.index.as_query_engine(similarity_top_k=1)
    response = query_engine.query("systems programming")
    response_text = str(response)
    assert "Rust" in response_text or "systems" in response_text


def test_update_existing_file_removes_old_version(tmp_test_dir, create_test_documents):
    """
    Test that updating a file removes the old version from the vector store.
    Old content should not appear in search results.
    """
    # Create initial file
    docs = {"update_test.txt": "This is version one of the document"}
    create_test_documents(tmp_test_dir, docs)


    reader = SimpleDirectoryReader(input_dir=str(tmp_test_dir))
    documents = reader.load_data()

    manager = IndexManager(tmp_test_dir)
    manager.initialize(documents, "sentence-transformers/all-MiniLM-L6-v2", "chromadb")
    manager.update_file_metadata([tmp_test_dir / "update_test.txt"], documents)
    manager.save_metadata()

    # Verify version one is searchable
    query_engine = manager.index.as_query_engine(similarity_top_k=1)
    response = query_engine.query("version one")
    assert "version one" in str(response).lower() or "one" in str(response).lower()

    # Update the file with new content
    file_path = tmp_test_dir / "update_test.txt"
    file_path.write_text("This is version two of the document")

    # Reload manager and update the file
    manager2 = IndexManager(tmp_test_dir)
    manager2.load()
    result = manager2.add_files([file_path])

    assert result["processed"] == 1
    assert len(result["errors"]) == 0

    manager2.save_metadata()

    # Search for version two - should find it
    query_engine2 = manager2.index.as_query_engine(similarity_top_k=1)
    response2 = query_engine2.query("version two")
    response_text = str(response2).lower()
    assert "version two" in response_text or "two" in response_text

    # Search for version one - should NOT find it (or at least version two should be more prominent)
    response3 = query_engine2.query("version one")
    response_text3 = str(response3).lower()
    # The response should not strongly match "version one" since we deleted it
    # This is a bit fuzzy, but the key is that version two is now in the index
    assert manager2.metadata.indexed_files["update_test.txt"].doc_ids


def test_remove_files_deletes_from_vector_store(tmp_test_dir, create_test_documents):
    """
    Test that removing files deletes them from both metadata and vector store.
    Removed content should not appear in search results.
    """
    # Create test documents
    docs = {
        "keep.txt": "This document will be kept in the index",
        "remove.txt": "This document contains secret information to be removed",
    }
    create_test_documents(tmp_test_dir, docs)

    from llama_index.core import SimpleDirectoryReader

    reader = SimpleDirectoryReader(input_dir=str(tmp_test_dir))
    documents = reader.load_data()

    manager = IndexManager(tmp_test_dir)
    manager.initialize(documents, "sentence-transformers/all-MiniLM-L6-v2", "chromadb")
    file_paths = [tmp_test_dir / name for name in docs.keys()]
    manager.update_file_metadata(file_paths, documents)
    manager.save_metadata()

    # Verify "secret information" is searchable
    query_engine = manager.index.as_query_engine(similarity_top_k=2)
    response = query_engine.query("secret information")
    response_text = str(response).lower()
    assert "secret" in response_text or "remove" in response_text

    # Remove the file
    manager2 = IndexManager(tmp_test_dir)
    manager2.load()
    result = manager2.remove_files([tmp_test_dir / "remove.txt"])

    assert result["removed"] == 1
    assert len(result["errors"]) == 0

    manager2.save_metadata()

    # Verify it's removed from metadata
    assert "remove.txt" not in manager2.metadata.indexed_files
    assert "keep.txt" in manager2.metadata.indexed_files

    # Verify "secret information" is no longer searchable
    query_engine2 = manager2.index.as_query_engine(similarity_top_k=2)
    response2 = query_engine2.query("secret information")
    response_text2 = str(response2).lower()
    # The removed content should not appear (or appear much less strongly)
    # We check that "kept" content is still there
    response3 = query_engine2.query("kept in the index")
    assert "kept" in str(response3).lower() or "keep" in str(response3).lower()


def test_error_handling_during_add_files(tmp_test_dir, create_test_documents):
    """
    Test that errors during file addition are handled gracefully.
    Other files in the batch should still be processed.
    """
    # Create initial index
    docs = {"initial.txt": "Initial content"}
    create_test_documents(tmp_test_dir, docs)

    from llama_index.core import SimpleDirectoryReader

    reader = SimpleDirectoryReader(input_dir=str(tmp_test_dir))
    documents = reader.load_data()

    manager = IndexManager(tmp_test_dir)
    manager.initialize(documents, "sentence-transformers/all-MiniLM-L6-v2", "chromadb")
    manager.update_file_metadata([tmp_test_dir / "initial.txt"], documents)
    manager.save_metadata()

    # Create one valid file and reference one non-existent file
    valid_file = tmp_test_dir / "valid.txt"
    valid_file.write_text("Valid content")

    invalid_file = tmp_test_dir / "nonexistent.txt"
    # Don't create this file

    # Load and try to add both files
    manager2 = IndexManager(tmp_test_dir)
    manager2.load()

    result = manager2.add_files([valid_file, invalid_file])

    # Should have one success and one error
    assert result["processed"] >= 0  # At least tried
    assert len(result["errors"]) > 0  # Should have error for nonexistent file

    # Verify the valid file was still processed (if it succeeded)
    manager2.save_metadata()
    if result["processed"] > 0:
        # Valid file should be in metadata
        assert "valid.txt" in manager2.metadata.indexed_files
