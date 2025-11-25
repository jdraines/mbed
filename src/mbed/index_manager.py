import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import chromadb
from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from .loaders import load_index
from .metadata import MetadataManager
from .utils import make_mbed_dir

logger = logging.getLogger(__name__)

StorageType = Literal["chromadb", "simple"]


class IndexManager:
    """Manages vector index operations for a directory."""

    def __init__(self, directory: Path):
        """
        Initialize IndexManager for a directory.

        Args:
            directory: Root directory being indexed
        """
        self.directory = directory
        self.mbed_dir = directory / ".mbed"
        self.metadata_mgr = MetadataManager(self.mbed_dir)

        self._index = None
        self._embed_model = None
        self._metadata = None

    def load(self) -> None:
        """Load existing index and metadata."""
        if not self.mbed_dir.exists():
            raise ValueError(f"Directory {self.directory} is not indexed.")

        logger.info("Loading existing index")
        self._metadata = self.metadata_mgr.load_metadata()

        model_name = self._metadata["model_name"]

        # Use Settings.embed_model if already configured (e.g., mock in tests)
        # Check _embed_model directly to avoid triggering lazy initialization
        if hasattr(Settings, "_embed_model") and Settings._embed_model is not None:
            logger.debug("Using configured embed_model from Settings")
            self._embed_model = Settings._embed_model
        else:
            logger.debug(f"Loading embedding model {model_name}")
            self._embed_model = HuggingFaceEmbedding(model_name=model_name)

        logger.debug(f"Loading index for dir {self.mbed_dir}")
        self._index = load_index(self.mbed_dir, self._metadata, self._embed_model)

    def initialize(
        self,
        documents: list,
        model_name: str,
        storage_type: StorageType = "chromadb",
        top_k: int = 3,
        exclude: list[str] | None = None,
    ) -> None:
        """
        Create a new index with initial documents.

        Args:
            documents: List of documents to index
            model_name: HuggingFace embedding model name
            storage_type: Vector storage backend type ("chromadb" or "simple")
            top_k: Default number of search results
            exclude: List of file patterns to exclude from indexing

        Raises:
            ValueError: If directory is already indexed
        """
        if self.metadata_mgr.metadata_exists():
            raise ValueError(
                f"Directory {self.directory} is already indexed. "
                "Use load() for existing indexes."
            )

        # Create .mbed directory
        make_mbed_dir(self.directory)

        logger.info(f"Initializing index with model: {model_name}, storage: {storage_type}")

        # Initialize embedding model
        # Use Settings.embed_model if already configured (e.g., mock in tests)
        # Check _embed_model directly to avoid triggering lazy initialization
        # Otherwise create a new HuggingFaceEmbedding
        if hasattr(Settings, "_embed_model") and Settings._embed_model is not None:
            logger.info("Using configured embed_model from Settings")
            self._embed_model = Settings._embed_model
        else:
            logger.info(f"Loading embedding model: {model_name}")
            self._embed_model = HuggingFaceEmbedding(model_name=model_name)

        # Create vector store based on storage type
        if storage_type == "simple":
            logger.info("Creating SimpleVectorStore (in-memory)")
            logger.info("Building index")
            self._index = VectorStoreIndex.from_documents(
                documents,
                embed_model=self._embed_model,
                show_progress=True,
            )
        elif storage_type == "chromadb":
            logger.info("Creating ChromaDB vector store")
            chroma_client = chromadb.PersistentClient(
                path=str(self.mbed_dir / "chroma_db")
            )
            chroma_collection = chroma_client.get_or_create_collection("documents")
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

            storage_context = StorageContext.from_defaults(vector_store=vector_store)

            logger.info("Building index")
            self._index = VectorStoreIndex.from_documents(
                documents,
                storage_context=storage_context,
                embed_model=self._embed_model,
                show_progress=True,
            )
        else:
            raise ValueError(f"Unsupported storage type: {storage_type}")

        # Initialize metadata
        self._metadata = {
            "model_name": model_name,
            "storage_type": storage_type,
            "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "last_updated": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "indexed_files": {},
            "config": {"top_k": top_k, "exclude": exclude or []},
        }

    def add_files(self, file_paths: list[Path]) -> dict:
        """
        Add or update files in the index.

        Args:
            file_paths: List of file paths to add/update

        Returns:
            Dictionary with:
            - processed: Number of files successfully processed
            - errors: List of (file_path, error_message) tuples

        Raises:
            ValueError: If index is not loaded or initialized
        """
        if self._index is None or self._metadata is None:
            raise ValueError("Index not loaded. Call load() or initialize() first.")

        logger.info(f"Adding {len(file_paths)} files to index")

        errors = []
        processed = 0

        for file_path in file_paths:
            try:
                # If file was previously indexed, delete old version first
                rel_path = str(file_path.relative_to(self.directory))
                if rel_path in self._metadata["indexed_files"]:
                    old_doc_ids = self._metadata["indexed_files"][rel_path].get(
                        "doc_ids", []
                    )
                    for doc_id in old_doc_ids:
                        try:
                            self._index.delete_ref_doc(doc_id, delete_from_docstore=True)
                            logger.debug(f"Deleted old doc_id {doc_id} for {file_path.name}")
                        except Exception as e:
                            logger.warning(f"Could not delete old doc_id {doc_id}: {e}")

                # Read and insert new documents
                reader = SimpleDirectoryReader(input_files=[str(file_path)])
                documents = reader.load_data()

                doc_ids = []
                for doc in documents:
                    self._index.insert(doc)
                    # Capture the document ID (llama-index assigns this)
                    doc_ids.append(doc.doc_id)

                # Update metadata with new doc IDs
                stat = file_path.stat()
                self._metadata["indexed_files"][rel_path] = {
                    "path": str(file_path),
                    "mtime": stat.st_mtime,
                    "size": stat.st_size,
                    "doc_ids": doc_ids,
                    "indexed_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                }
                processed += 1
                logger.debug(f"Indexed {file_path.name} with {len(doc_ids)} doc(s)")
            except Exception as e:
                error_msg = str(e)
                errors.append((file_path, error_msg))
                logger.error(f"Error indexing {file_path}: {error_msg}")

        # Update last_updated timestamp
        self._metadata["last_updated"] = datetime.now(UTC).isoformat().replace("+00:00", "Z")

        return {
            "processed": processed,
            "errors": errors,
        }

    def update_file_metadata(self, file_paths: list[Path], documents: list = None) -> None:
        """
        Update metadata for files (useful after initial indexing).

        Args:
            file_paths: List of file paths to track in metadata
            documents: Optional list of documents (to extract doc_ids)
        """
        if self._metadata is None:
            raise ValueError("Index not loaded. Call load() or initialize() first.")

        # Build a map of file_path to document for doc_id extraction
        doc_map = {}
        if documents:
            for doc in documents:
                file_path = Path(doc.metadata.get("file_path", ""))
                if file_path not in doc_map:
                    doc_map[file_path] = []
                doc_map[file_path].append(doc)

        for file_path in file_paths:
            if file_path.exists():
                stat = file_path.stat()
                rel_path = str(file_path.relative_to(self.directory))

                # Extract doc_ids if available
                doc_ids = []
                if file_path in doc_map:
                    doc_ids = [doc.doc_id for doc in doc_map[file_path]]

                self._metadata["indexed_files"][rel_path] = {
                    "path": str(file_path),
                    "mtime": stat.st_mtime,
                    "size": stat.st_size,
                    "doc_ids": doc_ids,
                    "indexed_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                }

    def remove_files(self, file_paths: list[Path]) -> dict:
        """
        Remove files from both the vector store and metadata.

        Args:
            file_paths: List of file paths to remove

        Returns:
            Dictionary with:
            - removed: Number of files successfully removed
            - errors: List of (file_path, error_message) tuples
        """
        if self._index is None or self._metadata is None:
            raise ValueError("Index not loaded. Call load() or initialize() first.")

        logger.info(f"Removing {len(file_paths)} files from index")

        errors = []
        removed = 0

        for file_path in file_paths:
            try:
                rel_path = str(file_path.relative_to(self.directory))
                if rel_path in self._metadata["indexed_files"]:
                    # Get document IDs and delete from vector store
                    doc_ids = self._metadata["indexed_files"][rel_path].get("doc_ids", [])

                    for doc_id in doc_ids:
                        try:
                            self._index.delete_ref_doc(doc_id, delete_from_docstore=True)
                            logger.debug(f"Deleted doc_id {doc_id} for {file_path.name}")
                        except Exception as e:
                            logger.warning(f"Could not delete doc_id {doc_id}: {e}")

                    # Remove from metadata
                    del self._metadata["indexed_files"][rel_path]
                    removed += 1
                    logger.debug(f"Removed {file_path.name} from index")
                else:
                    logger.warning(f"File {rel_path} not found in metadata")
            except Exception as e:
                error_msg = str(e)
                errors.append((file_path, error_msg))
                logger.error(f"Error removing {file_path}: {error_msg}")

        self._metadata["last_updated"] = datetime.now(UTC).isoformat().replace("+00:00", "Z")

        return {
            "removed": removed,
            "errors": errors,
        }

    def remove_file_metadata(self, file_paths: list[Path]) -> None:
        """
        Remove files from metadata only (does not remove from vector store).
        Deprecated: Use remove_files() for proper deletion.

        Args:
            file_paths: List of file paths to remove from metadata
        """
        if self._metadata is None:
            raise ValueError("Index not loaded. Call load() or initialize() first.")

        logger.warning("Using deprecated remove_file_metadata(). Use remove_files() instead.")

        for file_path in file_paths:
            rel_path = str(file_path.relative_to(self.directory))
            if rel_path in self._metadata["indexed_files"]:
                del self._metadata["indexed_files"][rel_path]

        self._metadata["last_updated"] = datetime.now(UTC).isoformat().replace("+00:00", "Z")

    def save_metadata(self) -> None:
        """Save current metadata to disk."""
        if self._metadata is None:
            raise ValueError("No metadata to save.")

        logger.info("Saving metadata")
        self.metadata_mgr.save_metadata(self._metadata)

    @property
    def index(self):
        """Get the vector store index."""
        return self._index

    @property
    def metadata(self):
        """Get the current metadata."""
        return self._metadata
