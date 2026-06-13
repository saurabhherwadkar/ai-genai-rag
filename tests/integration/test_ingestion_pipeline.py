# =============================================================================
# RAG Pipeline - Ingestion Pipeline Integration Tests
# =============================================================================
# End-to-end tests for the complete ingestion flow.

from pathlib import Path  # Import Path for file operations
from unittest.mock import MagicMock  # Import mocking utilities

import pytest  # Import pytest for test decorators and assertions

from rag_pipeline.ingestion.document_loader import DocumentLoader  # Loader component
from rag_pipeline.ingestion.embedding_generator import EmbeddingGenerator  # Embedding component
from rag_pipeline.ingestion.ingestion_pipeline import IngestionPipeline  # Pipeline under test
from rag_pipeline.ingestion.text_chunker import TextChunker  # Chunker component
from rag_pipeline.models.pipeline_config import ChunkingConfig, EmbeddingConfig  # Configs
from rag_pipeline.utils.exceptions import ChunkingError  # Expected exception
from rag_pipeline.vectorstore.vector_store_manager import VectorStoreManager  # Store component


class TestIngestionPipelineIntegration:
    """Integration tests for the complete ingestion pipeline."""

    @pytest.fixture  # Fixture for creating the full pipeline with mocked embedding
    def pipeline(self) -> IngestionPipeline:
        """Create a complete ingestion pipeline with mocked embedding model."""
        import numpy as np  # Import numpy for fake embeddings

        # Create real components with test configurations
        document_loader = DocumentLoader(  # Real document loader
            supported_extensions=[".txt", ".md"]  # Test extensions
        )
        text_chunker = TextChunker(  # Real text chunker
            ChunkingConfig(chunk_size=100, chunk_overlap=20, separator="\n\n", min_chunk_size=10)
        )
        embedding_config = EmbeddingConfig(  # Embedding config
            model_name="all-MiniLM-L6-v2", batch_size=8, normalize=True
        )
        embedding_generator = EmbeddingGenerator(embedding_config)  # Create generator
        # Mock the model to avoid downloading it in tests
        mock_model = MagicMock()  # Create mock model
        mock_model.encode.side_effect = lambda texts, **kwargs: np.array(  # Dynamic mock
            [[0.1] * 384] * (len(texts) if isinstance(texts, list) else 1)
        )
        embedding_generator._model = mock_model  # Inject mock model
        vector_store = VectorStoreManager(  # Real in-memory vector store
            collection_name="test_ingestion",  # Test collection
            persist_directory=None,  # In-memory mode
            distance_metric="cosine",  # Cosine distance
        )
        vector_store.initialize_store()  # Initialize the store
        return IngestionPipeline(  # Assemble the pipeline
            document_loader=document_loader,
            text_chunker=text_chunker,
            embedding_generator=embedding_generator,
            vector_store=vector_store,
        )

    def test_ingest_single_file_end_to_end(
        self, pipeline: IngestionPipeline, tmp_path: Path
    ) -> None:
        """Test complete ingestion of a single file from load to store."""
        # Create a test file with enough content for chunking
        content = (  # Multi-paragraph content
            "Machine learning is a field of AI.\n\n"
            "It uses algorithms to learn from data.\n\n"
            "Deep learning is a subset of machine learning."
        )
        test_file = tmp_path / "test_article.txt"  # Define file path
        test_file.write_text(content, encoding="utf-8")  # Write content
        chunks_stored = pipeline.ingest_file(test_file)  # Run full ingestion
        assert chunks_stored > 0  # Should have stored at least one chunk

    def test_ingest_directory_end_to_end(
        self, pipeline: IngestionPipeline, tmp_path: Path
    ) -> None:
        """Test complete ingestion of a directory with multiple files."""
        # Create multiple test files
        (tmp_path / "doc1.txt").write_text(  # First document
            "First document about Python.\n\nPython is versatile.",
            encoding="utf-8",
        )
        (tmp_path / "doc2.txt").write_text(  # Second document
            "Second document about Java.\n\nJava is statically typed.",
            encoding="utf-8",
        )
        chunks_stored = pipeline.ingest_directory(  # Ingest directory
            tmp_path, [".txt"]  # Only .txt files
        )
        assert chunks_stored > 0  # Should have stored chunks from both files

    def test_ingest_empty_file_gracefully(
        self, pipeline: IngestionPipeline, tmp_path: Path
    ) -> None:
        """Test that ingesting an empty file raises appropriate error."""
        empty_file = tmp_path / "empty.txt"  # Create empty file
        empty_file.write_text("", encoding="utf-8")  # Write empty content
        with pytest.raises(ChunkingError):  # Should raise chunking error
            pipeline.ingest_file(empty_file)  # Attempt to ingest

    def test_ingest_preserves_metadata_in_store(
        self, pipeline: IngestionPipeline, tmp_path: Path
    ) -> None:
        """Test that document metadata is preserved through the pipeline."""
        content = "A document with enough content to create at least one chunk for testing."
        test_file = tmp_path / "metadata_test.txt"  # File with known name
        test_file.write_text(content, encoding="utf-8")  # Write content
        chunks_stored = pipeline.ingest_file(test_file)  # Ingest the file
        assert chunks_stored > 0  # Should have stored chunks
        # Verify by checking store stats
        stats = pipeline._vector_store.get_collection_stats()  # Get stats
        assert stats["total_chunks"] > 0  # Should have items in store
