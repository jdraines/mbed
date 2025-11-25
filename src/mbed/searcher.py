import logging
from pathlib import Path

from .index_manager import IndexManager

logger = logging.getLogger(__name__)


def search_directory(directory: Path, query: str, top_k: int | None = None):
    """
    Perform vector search in an indexed directory.

    Args:
        directory: Path to indexed directory
        query: Search query string
        top_k: Override number of results (uses metadata default if None)

    Returns:
        Query response with relevant documents
    """
    # Load index manager
    manager = IndexManager(directory)
    manager.load()

    if top_k is None:
        top_k = manager.metadata.get("config", {}).get("top_k", 3)

    logger.info(f"Searching for: {query}")

    # Perform query
    query_engine = manager.index.as_query_engine(similarity_top_k=top_k)
    response = query_engine.query(query)

    return response
