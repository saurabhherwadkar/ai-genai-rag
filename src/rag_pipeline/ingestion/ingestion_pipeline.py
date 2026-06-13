# =============================================================================
# RAG Pipeline - Ingestion Pipeline Orchestrator
# =============================================================================
# Orchestrates the full document ingestion flow: load -> chunk -> embed -> store.

import logging  # Import logging for pipeline execution tracking
from pathlib import Path  # Import Path for file system operations

from rag_pipeline.ingestion.document_loader import DocumentLoader  # Document loading component
from rag_pipeline.ingestion.embedding_generator import EmbeddingGenerator  # Embedding component
from rag_pipeline.ingestion.text_chunker import TextChunker  # Text chunking component
from rag_pipeline.utils.exceptions import RAGPipelineError  # Base pipeline exception
from rag_pipeline.utils.metrics import PerformanceMetrics  # Timing utilities
from rag_pipeline.vectorstore.vector_store_manager import VectorStoreManager  # Storage component

logger = logging.getLogger(__name__)  # Create module-level logger instance


class IngestionPipeline:
    """Orchestrates the full document ingestion flow.

    Coordinates the sequence: load documents -> chunk text -> generate embeddings
    -> store in vector database. Provides both single-file and directory ingestion.
    """

    def __init__(
        self,
        document_loader: DocumentLoader,
        text_chunker: TextChunker,
        embedding_generator: EmbeddingGenerator,
        vector_store: VectorStoreManager,
    ) -> None:
        """Initialize the ingestion pipeline with all required components.

        Args:
            document_loader: Component for loading documents from files.
            text_chunker: Component for splitting documents into chunks.
            embedding_generator: Component for generating vector embeddings.
            vector_store: Component for storing chunks in the vector database.
        """
        self._document_loader = document_loader  # Store document loader reference
        self._text_chunker = text_chunker  # Store text chunker reference
        self._embedding_generator = embedding_generator  # Store embedding generator reference
        self._vector_store = vector_store  # Store vector store manager reference
        self._logger = logger  # Store logger reference for this instance

    def ingest_file(self, file_path: Path) -> int:
        """Ingest a single file through the complete pipeline.

        Args:
            file_path: Path to the file to ingest.

        Returns:
            Number of chunks successfully stored in the vector database.

        Raises:
            RAGPipelineError: If any stage of the pipeline fails.
        """
        self._logger.info("Starting ingestion for file: %s", file_path.name)  # Log start
        with PerformanceMetrics.measure_time() as timing:  # Measure total ingestion time
            document = self._document_loader.load_file(file_path)  # Step 1: Load document
            chunks_stored = self._process_single_document(document)  # Steps 2-4: Process
        self._logger.info(  # Log completion with metrics
            "File ingestion complete: %s (%d chunks, %.1fms)",  # Format message
            file_path.name,  # Filename for identification
            chunks_stored,  # Number of chunks stored
            timing["elapsed_ms"],  # Total processing time
        )
        return chunks_stored  # Return count of stored chunks

    def ingest_directory(self, directory_path: Path, extensions: list[str]) -> int:
        """Ingest all supported files from a directory.

        Args:
            directory_path: Path to the directory to scan and ingest.
            extensions: List of file extensions to include in ingestion.

        Returns:
            Total number of chunks stored across all documents.

        Raises:
            RAGPipelineError: If directory loading or processing fails.
        """
        self._logger.info(  # Log directory ingestion start
            "Starting directory ingestion: %s (extensions=%s)",  # Format message
            directory_path,  # Directory path for identification
            extensions,  # Extensions being processed
        )
        with PerformanceMetrics.measure_time() as timing:  # Measure total directory time
            documents = self._document_loader.load_directory(  # Load all documents
                directory_path, extensions  # Pass directory and extension filter
            )
            total_chunks = 0  # Initialize total chunk counter
            for document in documents:  # Process each loaded document
                try:  # Attempt to process each document independently
                    chunks_stored = self._process_single_document(document)  # Process document
                    total_chunks += chunks_stored  # Accumulate chunk count
                except RAGPipelineError as error:  # Catch pipeline errors per document
                    self._logger.warning(  # Log warning but continue with other documents
                        "Failed to process document %s: %s",  # Format message
                        document.source_path,  # Identify the failed document
                        error,  # Include the error details
                    )
        self._log_ingestion_metrics(  # Log final summary metrics
            len(documents), total_chunks, timing["elapsed_ms"]  # Pass counts and timing
        )
        return total_chunks  # Return total chunks stored across all documents

    def _process_single_document(self, document) -> int:
        """Process a single document through chunk, embed, and store stages.

        Args:
            document: The Document object to process.

        Returns:
            Number of chunks stored for this document.
        """
        chunks = self._text_chunker.chunk_document(document)  # Step 2: Chunk the document
        self._logger.debug(  # Log chunking result
            "Document %s split into %d chunks",  # Format message
            document.document_id[:8],  # Document ID prefix
            len(chunks),  # Number of chunks created
        )
        embedded_chunks = self._embedding_generator.embed_chunks(chunks)  # Step 3: Embed chunks
        self._logger.debug(  # Log embedding result
            "Generated embeddings for %d chunks",  # Format message
            len(embedded_chunks),  # Number of embeddings generated
        )
        chunks_stored = self._vector_store.add_chunks(embedded_chunks)  # Step 4: Store chunks
        return chunks_stored  # Return number of chunks stored

    def _log_ingestion_metrics(
        self, doc_count: int, chunk_count: int, duration_ms: float
    ) -> None:
        """Log summary metrics for a batch ingestion operation.

        Args:
            doc_count: Number of documents processed.
            chunk_count: Total number of chunks stored.
            duration_ms: Total processing time in milliseconds.
        """
        throughput = PerformanceMetrics.calculate_throughput(  # Calculate documents per second
            doc_count, duration_ms  # Pass count and time
        )
        self._logger.info(  # Log comprehensive metrics summary
            "Ingestion complete: %d docs, %d chunks, %.1fms (%.1f docs/sec)",  # Format
            doc_count,  # Number of documents processed
            chunk_count,  # Total chunks stored
            duration_ms,  # Total time in milliseconds
            throughput,  # Processing speed
        )
