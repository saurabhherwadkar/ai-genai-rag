# =============================================================================
# RAG Pipeline - Retriever Unit Tests
# =============================================================================
# Tests for the Retriever class with mocked dependencies.

from unittest.mock import MagicMock  # Import MagicMock for dependency mocking

import pytest  # Import pytest for test decorators and assertions

from rag_pipeline.models.chunk import Chunk  # Import Chunk for creating test results
from rag_pipeline.models.pipeline_config import RetrievalConfig  # Import config model
from rag_pipeline.models.query_result import QueryResult  # Import result model
from rag_pipeline.query.retriever import Retriever  # Class under test


class TestRetriever:
    """Unit tests for the Retriever class."""

    @pytest.fixture  # Fixture for mock query results
    def mock_query_results(self) -> list[QueryResult]:
        """Create mock query results for testing."""
        return [  # Return list of mock results
            QueryResult(  # First result with high similarity
                chunk=Chunk(
                    chunk_id="r1",
                    document_id="doc-1",
                    content="Machine learning is great",
                    chunk_index=0,
                    start_char=0,
                    end_char=25,
                    metadata={"filename": "test.txt"},
                ),
                similarity_score=0.9,  # High similarity
                rank=1,  # First rank
            ),
            QueryResult(  # Second result with medium similarity
                chunk=Chunk(
                    chunk_id="r2",
                    document_id="doc-1",
                    content="Deep learning uses layers",
                    chunk_index=1,
                    start_char=26,
                    end_char=51,
                    metadata={"filename": "test.txt"},
                ),
                similarity_score=0.5,  # Medium similarity
                rank=2,  # Second rank
            ),
            QueryResult(  # Third result with low similarity
                chunk=Chunk(
                    chunk_id="r3",
                    document_id="doc-1",
                    content="Unrelated content here",
                    chunk_index=2,
                    start_char=52,
                    end_char=74,
                    metadata={"filename": "test.txt"},
                ),
                similarity_score=0.1,  # Below typical threshold
                rank=3,  # Third rank
            ),
        ]

    @pytest.fixture  # Fixture for creating a Retriever with mocked deps
    def retriever(
        self, retrieval_config: RetrievalConfig, mock_query_results: list[QueryResult]
    ) -> Retriever:
        """Create a Retriever with mocked vector store and embedding generator."""
        mock_store = MagicMock()  # Mock vector store
        mock_store.search_similar.return_value = mock_query_results  # Return test results
        mock_embedding_gen = MagicMock()  # Mock embedding generator
        mock_embedding_gen.generate_embedding.return_value = [0.1] * 384  # Fake embedding
        return Retriever(mock_store, mock_embedding_gen, retrieval_config)  # Create retriever

    def test_retrieve_returns_results(self, retriever: Retriever) -> None:
        """Test that retrieve returns a non-empty list of results."""
        results = retriever.retrieve("What is machine learning?")  # Execute retrieval
        assert isinstance(results, list)  # Should return a list
        assert len(results) > 0  # Should have at least one result

    def test_retrieve_filters_below_threshold(self, retriever: Retriever) -> None:
        """Test that results below similarity threshold are filtered out."""
        results = retriever.retrieve("What is machine learning?")  # Execute retrieval
        # With threshold 0.2, the result with score 0.1 should be filtered
        for result in results:  # Check each result
            assert result.similarity_score >= 0.2  # All should be above threshold

    def test_retrieve_deduplicates_content(self, retrieval_config: RetrievalConfig) -> None:
        """Test that duplicate content is removed from results."""
        duplicate_results = [  # Results with duplicate content
            QueryResult(
                chunk=Chunk(
                    chunk_id="dup1", document_id="d1", content="Same content here",
                    chunk_index=0, start_char=0, end_char=17, metadata={},
                ),
                similarity_score=0.8, rank=1,
            ),
            QueryResult(
                chunk=Chunk(
                    chunk_id="dup2", document_id="d1", content="Same content here",
                    chunk_index=1, start_char=18, end_char=35, metadata={},
                ),
                similarity_score=0.7, rank=2,
            ),
        ]
        mock_store = MagicMock()  # Mock store
        mock_store.search_similar.return_value = duplicate_results  # Return duplicates
        mock_embedding_gen = MagicMock()  # Mock embedding generator
        mock_embedding_gen.generate_embedding.return_value = [0.1] * 384  # Fake embedding
        retriever = Retriever(mock_store, mock_embedding_gen, retrieval_config)  # Create
        results = retriever.retrieve("test query")  # Execute retrieval
        assert len(results) == 1  # Duplicates should be removed

    def test_retrieve_calls_embedding_generator(
        self, retrieval_config: RetrievalConfig
    ) -> None:
        """Test that the query is embedded before searching."""
        mock_store = MagicMock()  # Mock store
        mock_store.search_similar.return_value = []  # Empty results
        mock_embedding_gen = MagicMock()  # Mock embedding generator
        mock_embedding_gen.generate_embedding.return_value = [0.1] * 384  # Fake
        retriever = Retriever(mock_store, mock_embedding_gen, retrieval_config)  # Create
        retriever.retrieve("test query")  # Execute retrieval
        mock_embedding_gen.generate_embedding.assert_called_once_with("test query")  # Verify
