# =============================================================================
# RAG Pipeline - Data Models Package
# =============================================================================
# Exports all data model classes used throughout the pipeline.

from rag_pipeline.models.chunk import Chunk  # Chunk data model for text segments
from rag_pipeline.models.document import Document  # Document data model for loaded files
from rag_pipeline.models.pipeline_config import (  # Configuration data models
    ChunkingConfig,
    EmbeddingConfig,
    RetrievalConfig,
)
from rag_pipeline.models.query_result import QueryResponse, QueryResult  # Query result models

__all__ = [  # Public API of the models package
    "Document",
    "Chunk",
    "QueryResult",
    "QueryResponse",
    "ChunkingConfig",
    "EmbeddingConfig",
    "RetrievalConfig",
]
