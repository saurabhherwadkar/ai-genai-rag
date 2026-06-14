# =============================================================================
# RAG Pipeline - Pipeline Configuration Data Models
# =============================================================================
# Defines typed configuration dataclasses for each pipeline component.

from dataclasses import dataclass  # Import dataclass decorator for typed configs


@dataclass(frozen=True)  # Frozen ensures config immutability after creation
class ChunkingConfig:
    """Configuration parameters for the text chunking strategy.

    Controls how documents are split into smaller pieces for embedding.
    """

    chunk_size: int = 512  # Maximum number of characters per chunk
    chunk_overlap: int = 50  # Number of overlapping characters between adjacent chunks
    separator: str = "\n\n"  # Primary separator to split text on (paragraph boundary)
    min_chunk_size: int = 50  # Minimum characters required to keep a chunk


@dataclass(frozen=True)  # Frozen ensures config immutability after creation
class EmbeddingConfig:
    """Configuration parameters for embedding generation.

    Controls the model used and batch processing behavior.
    """

    model_name: str = "all-MiniLM-L6-v2"  # Sentence-transformer model identifier
    batch_size: int = 32  # Number of texts to embed in a single batch
    normalize: bool = True  # Whether to L2-normalize embeddings to unit vectors


@dataclass(frozen=True)  # Frozen ensures config immutability after creation
class RetrievalConfig:
    """Configuration parameters for the retrieval and reranking step.

    Controls how many results are fetched and quality thresholds.
    """

    top_k: int = 5  # Maximum number of results to return from similarity search
    similarity_threshold: float = 0.3  # Minimum similarity score to include a result
    rerank_enabled: bool = True  # Whether to apply reranking after initial retrieval
    max_query_length: int = 1000  # Maximum allowed characters in a query string
    hybrid_search_enabled: bool = True  # Whether to use hybrid (semantic + BM25) retrieval
    semantic_top_k: int = 50  # Number of results from semantic search before fusion
    bm25_top_k: int = 50  # Number of results from BM25 search before fusion
