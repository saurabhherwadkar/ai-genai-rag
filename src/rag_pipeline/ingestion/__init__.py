# =============================================================================
# RAG Pipeline - Ingestion Package
# =============================================================================
# Exports all ingestion pipeline components for document processing.

from rag_pipeline.ingestion.document_loader import DocumentLoader  # Document loading class
from rag_pipeline.ingestion.embedding_generator import EmbeddingGenerator  # Embedding class
from rag_pipeline.ingestion.ingestion_pipeline import IngestionPipeline  # Pipeline orchestrator
from rag_pipeline.ingestion.text_chunker import TextChunker  # Text chunking class

__all__ = [  # Public API of the ingestion package
    "DocumentLoader",
    "TextChunker",
    "EmbeddingGenerator",
    "IngestionPipeline",
]
