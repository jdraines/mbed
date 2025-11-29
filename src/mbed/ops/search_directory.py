import logging
from dataclasses import dataclass
from pathlib import Path

from ..index_manager import IndexManager

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SearchResult:
    """A search result with file path and similarity score."""

    file_path: str
    score: float


def search_directory(directory: Path, query: str, top_k: int | None = None) -> list[SearchResult]:
    """
    Perform vector search in an indexed directory.

    Args:
        directory: Path to indexed directory
        query: Search query string
        top_k: Override number of results (uses metadata default if None)

    Returns:
        Deduplicated list of SearchResult objects with file_path and similarity score
    """
    # Load index manager
    manager = IndexManager(directory)
    manager.load()

    if top_k is None:
        top_k = manager.metadata.config.top_k

    logger.info(f"Searching for: {query}")

    # Perform retrieval
    retriever = manager.index.as_retriever(similarity_top_k=top_k)
    nodes = retriever.retrieve(query)

    # Extract file paths and scores, deduplicate by file_path
    seen = {}
    for node_with_score in nodes:
        file_path = node_with_score.node.metadata.get("file_path")
        score = node_with_score.score
        if file_path:
            # Keep the highest score for each file_path
            if file_path not in seen or score > seen[file_path]:
                seen[file_path] = score

    # Convert to list of SearchResult objects
    results = [SearchResult(file_path=path, score=score) for path, score in seen.items()]

    return results
