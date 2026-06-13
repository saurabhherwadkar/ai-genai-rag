# =============================================================================
# RAG Pipeline - Query Processor Unit Tests
# =============================================================================
# Tests for the QueryProcessor class covering validation and normalization.

import pytest  # Import pytest for test decorators and assertions

from rag_pipeline.models.pipeline_config import RetrievalConfig  # Import config model
from rag_pipeline.query.query_processor import QueryProcessor  # Class under test
from rag_pipeline.security.input_sanitizer import InputSanitizer  # Dependency
from rag_pipeline.utils.exceptions import (  # Expected exceptions
    QueryValidationError,
    SecurityViolationError,
)


class TestQueryProcessor:
    """Unit tests for the QueryProcessor class."""

    @pytest.fixture  # Fixture for creating a QueryProcessor instance
    def processor(self, retrieval_config: RetrievalConfig) -> QueryProcessor:
        """Create a QueryProcessor with test configuration."""
        sanitizer = InputSanitizer(max_input_length=1000)  # Create sanitizer
        return QueryProcessor(retrieval_config, sanitizer)  # Create processor

    def test_process_query_returns_cleaned_string(
        self, processor: QueryProcessor
    ) -> None:
        """Test that process_query returns a non-empty cleaned string."""
        result = processor.process_query("What is machine learning?")  # Process query
        assert isinstance(result, str)  # Should return a string
        assert len(result) > 0  # Should not be empty

    def test_process_query_normalizes_whitespace(
        self, processor: QueryProcessor
    ) -> None:
        """Test that multiple whitespace characters are collapsed."""
        result = processor.process_query("What   is   machine   learning?")  # Extra spaces
        assert "   " not in result  # Should not have triple spaces
        assert "  " not in result  # Should not have double spaces

    def test_process_query_strips_leading_trailing(
        self, processor: QueryProcessor
    ) -> None:
        """Test that leading and trailing whitespace is removed."""
        result = processor.process_query("  What is ML?  ")  # Padded with spaces
        assert not result.startswith(" ")  # No leading space
        assert not result.endswith(" ")  # No trailing space

    def test_process_query_raises_for_empty_input(
        self, processor: QueryProcessor
    ) -> None:
        """Test that empty string input raises an error."""
        with pytest.raises(SecurityViolationError):  # Expect security error
            processor.process_query("")  # Pass empty string

    def test_process_query_raises_for_whitespace_only(
        self, processor: QueryProcessor
    ) -> None:
        """Test that whitespace-only input raises an error."""
        with pytest.raises(SecurityViolationError):  # Expect security error
            processor.process_query("   \t\n   ")  # Pass whitespace only

    def test_process_query_raises_for_exceeding_max_length(self) -> None:
        """Test that queries exceeding max length raise QueryValidationError."""
        config = RetrievalConfig(max_query_length=50)  # Very short max length
        sanitizer = InputSanitizer(max_input_length=5000)  # Sanitizer with higher limit
        processor = QueryProcessor(config, sanitizer)  # Create processor
        long_query = "a" * 100  # Query exceeding max length
        with pytest.raises(QueryValidationError):  # Expect validation error
            processor.process_query(long_query)  # Process overly long query

    def test_process_query_preserves_meaningful_content(
        self, processor: QueryProcessor
    ) -> None:
        """Test that meaningful query content is preserved after processing."""
        result = processor.process_query("How does deep learning work?")  # Standard query
        assert "deep" in result  # Key term preserved
        assert "learning" in result  # Key term preserved
        assert "work" in result  # Key term preserved

    def test_process_query_removes_special_characters(
        self, processor: QueryProcessor
    ) -> None:
        """Test that exotic special characters are removed."""
        result = processor.process_query("What is ML? @#$%^& test")  # With special chars
        assert "@" not in result  # Special char removed
        assert "#" not in result  # Special char removed
        assert "^" not in result  # Special char removed
