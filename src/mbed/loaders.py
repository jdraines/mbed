from pathlib import Path

import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore


def load_index(mbed_dir: Path, metadata: dict, embed_model):
    """
    Load index based on storage type in metadata.

    Args:
        mbed_dir: Path to .mbed directory
        metadata: Metadata dictionary with storage_type
        embed_model: Embedding model instance

    Returns:
        VectorStoreIndex instance
    """
    storage_type = metadata.storage_type

    if storage_type == "chromadb":
        chroma_client = chromadb.PersistentClient(path=str(mbed_dir / "chroma_db"))
        chroma_collection = chroma_client.get_collection("documents")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

        return VectorStoreIndex.from_vector_store(
            vector_store=vector_store, embed_model=embed_model
        )
    else:
        raise ValueError(f"Unsupported storage type: {storage_type}")
