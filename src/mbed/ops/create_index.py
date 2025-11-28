import logging
from pathlib import Path
from typing import Literal

from llama_index.core import SimpleDirectoryReader

from ..index_manager import IndexManager

logger = logging.getLogger(__name__)

StorageType = Literal["chromadb", "simple"]


def create_index(
    directory: Path,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    storage_type: StorageType = "chromadb",
    top_k: int = 3,
    exclude: list[str] | None = None,
):
    """
    Create initial index of all documents in a directory.

    Args:
        directory: Path to directory to index
        model_name: HuggingFace embedding model name
        storage_type: "chromadb" or "simple" (in-memory)
        top_k: Number of results to return in searches

    Returns:
        IndexManager instance
    """
    logger.info(f"Indexing directory: {directory}")
    logger.info(f"Using model: {model_name}, storage: {storage_type}")

    exclude = exclude or []
    exclude.append(".mbed")

    # Load documents
    reader = SimpleDirectoryReader(
        input_dir=str(directory), recursive=True, exclude=exclude
    )
    documents = reader.load_data()

    if not documents:
        raise ValueError(f"No documents found in {directory}")

    logger.info(f"Found {len(documents)} documents")

    # Collect file paths for metadata tracking
    file_paths = []
    for doc in documents:
        file_path = Path(doc.metadata.get("file_path", ""))
        if file_path.exists():
            file_paths.append(file_path)

    # Create and initialize index manager
    manager = IndexManager(directory)
    manager.initialize(documents, model_name, storage_type, top_k, exclude)

    # Update metadata with indexed files (pass documents to capture doc_ids)
    manager.update_file_metadata(file_paths, documents)
    manager.save_metadata()

    logger.info(
        f"Index created successfully at {directory / '.mbed'}, indexed {len(file_paths)} files"
    )

    return manager

