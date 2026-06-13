# =============================================================================
# RAG Pipeline - Reranker Unit Tests
# =============================================================================
# Tests for the Reranker class covering scoring and sorting logic.

import pytest  # Import pytest for test decorators and assertions

from rag_pipeline.models.chunk import Chunk  # Import Chunk for test data
from rag_pipeline.models.pipeline_config import RetrievalConfig  # Import config model
from rag_pipeline.models.query_result import QueryResult  # Import result model
from rag_pipeline.query.reranker import Reranker  # Class under test


class TestReranker:
    """Unit tests for the Reranker class."""

    @pytest.fixture  # Fixture for creating a Reranker instance
    def reranker(self, retrieval_config: RetrievalConfig) -> Reranker:
        """Create a Reranker with test configuration."""
        return Reranker(retrieval_config)  # Create with test config

    @pytest.fixture  # Fixture for test results to rerank
    def test_results(self) -> list[QueryResult]:
        """Create test QueryResult objects for reranking tests."""
        return [  # Return list of test results
            QueryResult(  # Result with high similarity but low keyword overlap
                chunk=Chunk(
                    chunk_id="c1", document_id="d1",
                    content="Neural networks are computational models inspired by biology",
                    chunk_index=0, start_char=0, end_char=60, metadata={},
                ),
                similarity_score=0.8,  # High vector similarity
                rank=1,  # Original rank 1
            ),
            QueryResult(  # Result with lower similarity but high keyword overlap
                chunk=Chunk(
                    chunk_id="c2", document_id="d1",
                    content="Machine learning algorithms learn patterns from data",
                    chunk_index=1, start_char=61, end_char=113, metadata={},
                ),
                similarity_score=0.6,  # Lower vector similarity
                rank=2,  # Original rank 2
            ),
            QueryResult(  # Result with lowest scores
                chunk=Chunk(
                    chunk_id="c3", document_id="d1",
                    content="Programming languages include Python and Java",
                    chunk_index=2, start_char=114, end_char=160, metadata={},
                ),
                similarity_score=0.4,  # Low vector similarity
                rank=3,  # Original rank 3
            ),
        ]

    def test_rerank_returns_same_count(
        self, reranker: Reranker, test_results: list[QueryResult]
    ) -> None:
        """Test that reranking preserves the number of results."""
        reranked = reranker.rerank("machine learning", test_results)  # Rerank
        assert len(reranked) == len(test_results)  # Same count

    def test_rerank_assigns_sequential_ranks(
        self, reranker: Reranker, test_results: list[QueryResult]
    ) -> None:
        """Test that reranked results have sequential rank numbers."""
        reranked = reranker.rerank("machine learning", test_results)  # Rerank
        ranks = [r.rank for r in reranked]  # Extract ranks
        assert ranks == [1, 2, 3]  # Should be sequential 1, 2, 3

    def test_rerank_empty_results(self, reranker: Reranker) -> None:
        """Test that reranking empty list returns empty list."""
        reranked = reranker.rerank("test query", [])  # Rerank empty list
        assert reranked == []  # Should return empty list

    def test_rerank_disabled_returns_original_order(
        self, test_results: list[QueryResult]
    ) -> None:
        """Test that disabled reranking returns results in original order."""
        config = RetrievalConfig(rerank_enabled=False)  # Disable reranking
        reranker = Reranker(config)  # Create reranker with disabled config
        reranked = reranker.rerank("test query", test_results)  # Rerank
        # Results should remain in original order
        assert reranked[0].chunk.chunk_id == "c1"  # First stays first
        assert reranked[2].chunk.chunk_id == "c3"  # Last stays last

    def test_keyword_overlap_boosts_matching_results(
        self, reranker: Reranker, test_results: list[QueryResult]
    ) -> None:
        """Test that results with keyword matches get boosted in ranking."""
        # Query containing "machine learning" should boost result c2
        reranked = reranker.rerank("machine learning algorithms", test_results)  # Rerank
        # The result about machine learning should rank highly
        top_content = reranked[0].chunk.content  # Get top result content
        assert "machine" in top_content.lower() or "learning" in top_content.lower()

    def test_compute_keyword_overlap_score_full_match(
        self, reranker: Reranker
    ) -> None:
        """Test keyword overlap returns 1.0 for full keyword match."""
        score = reranker._compute_keyword_overlap_score(  # Direct method test
            "machine learning",  # Query
            "Machine learning is a powerful technique",  # Text with all keywords
        )
        assert score == 1.0  # All keywords found

    def test_compute_keyword_overlap_score_no_match(
        self, reranker: Reranker
    ) -> None:
        """Test keyword overlap returns 0.0 for no keyword match."""
        score = reranker._compute_keyword_overlap_score(  # Direct method test
            "quantum physics",  # Query
            "The cat sat on the mat",  # Text with no matching keywords
        )
        assert score == 0.0  # No keywords found

    def test_compute_position_score_first_rank(self, reranker: Reranker) -> None:
        """Test that first rank gets maximum position score."""
        score = reranker._compute_position_score(1, 5)  # Rank 1 of 5
        assert score == 1.0  # First rank should get 1.0

    def test_compute_position_score_last_rank(self, reranker: Reranker) -> None:
        """Test that last rank gets minimum position score."""
        score = reranker._compute_position_score(5, 5)  # Rank 5 of 5
        assert score == 0.0  # Last rank should get 0.0
