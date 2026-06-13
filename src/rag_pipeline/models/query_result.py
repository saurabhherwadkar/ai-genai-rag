# =============================================================================
# RAG Pipeline - Query Result Data Models
# =============================================================================
# Defines QueryResult and QueryResponse classes for retrieval output.

from dataclasses import dataclass, field  # Import dataclass decorator and field factory

from rag_pipeline.models.chunk import Chunk  # Import Chunk for type reference


@dataclass  # Mutable dataclass since rank may be updated during reranking
class QueryResult:
    """Container for a single retrieval result with relevance scoring.

    Holds a reference to the retrieved chunk along with its similarity
    score and position rank in the result set.
    """

    chunk: Chunk  # The retrieved chunk from the vector store
    similarity_score: float  # Cosine similarity score between query and chunk
    rank: int  # Position in the result set after ranking (1-based)


@dataclass  # Mutable dataclass to allow progressive field population
class QueryResponse:
    """Complete response from the RAG pipeline for a user query.

    Aggregates the original query, retrieval results, generated answer,
    source attributions, and performance metrics.
    """

    original_query: str  # The user's original input query text
    processed_query: str  # The query after preprocessing and normalization
    results: list[QueryResult] = field(default_factory=list)  # Ranked retrieval results
    generated_answer: str = ""  # The final generated response text
    sources: list[str] = field(default_factory=list)  # Source file references for attribution
    latency_ms: float = 0.0  # End-to-end processing time in milliseconds
