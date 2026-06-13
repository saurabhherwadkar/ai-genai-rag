# =============================================================================
# RAG Pipeline - Response Generator Unit Tests
# =============================================================================
# Tests for the ResponseGenerator class covering answer generation.

import pytest  # Import pytest for test decorators and assertions

from rag_pipeline.generation.prompt_builder import PromptBuilder  # Dependency
from rag_pipeline.generation.response_generator import ResponseGenerator  # Class under test
from rag_pipeline.generation.template_engine import TemplateEngine  # Dependency
from rag_pipeline.models.chunk import Chunk  # Import Chunk for test data
from rag_pipeline.models.query_result import QueryResponse, QueryResult  # Result models


class TestResponseGenerator:
    """Unit tests for the ResponseGenerator class."""

    @pytest.fixture  # Fixture for template engine with test templates
    def template_engine(self) -> TemplateEngine:
        """Create a TemplateEngine with test response template."""
        templates = {  # Define test templates
            "response": "Answer for '{query}':\n{context}",  # Simple test template
        }
        return TemplateEngine(templates)  # Create and return engine

    @pytest.fixture  # Fixture for prompt builder
    def prompt_builder(self) -> PromptBuilder:
        """Create a PromptBuilder with test templates."""
        return PromptBuilder(  # Create with simple templates
            system_template="You are a helpful assistant.",  # System prompt
            user_template="Context:\n{context}\n\nQuestion: {query}",  # User prompt
        )

    @pytest.fixture  # Fixture for creating a ResponseGenerator
    def generator(
        self, prompt_builder: PromptBuilder, template_engine: TemplateEngine
    ) -> ResponseGenerator:
        """Create a ResponseGenerator with test dependencies."""
        return ResponseGenerator(  # Create generator
            prompt_builder=prompt_builder,  # Inject prompt builder
            template_engine=template_engine,  # Inject template engine
            response_max_length=500,  # Test max length
        )

    @pytest.fixture  # Fixture for test results
    def test_results(self) -> list[QueryResult]:
        """Create test retrieval results for response generation."""
        return [  # Return list of mock results
            QueryResult(  # Single result
                chunk=Chunk(
                    chunk_id="gen-1", document_id="doc-1",
                    content="Machine learning uses data to find patterns.",
                    chunk_index=0, start_char=0, end_char=45,
                    metadata={"filename": "ml_intro.txt"},
                ),
                similarity_score=0.85,  # High similarity
                rank=1,  # Top result
            ),
        ]

    def test_generate_response_returns_query_response(
        self, generator: ResponseGenerator, test_results: list[QueryResult]
    ) -> None:
        """Test that generate_response returns a QueryResponse object."""
        response = generator.generate_response("What is ML?", test_results)  # Generate
        assert isinstance(response, QueryResponse)  # Should return QueryResponse

    def test_generate_response_includes_answer(
        self, generator: ResponseGenerator, test_results: list[QueryResult]
    ) -> None:
        """Test that generated response contains a non-empty answer."""
        response = generator.generate_response("What is ML?", test_results)  # Generate
        assert len(response.generated_answer) > 0  # Answer should not be empty

    def test_generate_response_includes_sources(
        self, generator: ResponseGenerator, test_results: list[QueryResult]
    ) -> None:
        """Test that generated response includes source attributions."""
        response = generator.generate_response("What is ML?", test_results)  # Generate
        assert len(response.sources) > 0  # Should have at least one source
        assert "ml_intro.txt" in response.sources  # Should include the source file

    def test_generate_response_with_empty_results(
        self, generator: ResponseGenerator
    ) -> None:
        """Test response generation when no results are available."""
        response = generator.generate_response("Unknown topic", [])  # No results
        assert isinstance(response, QueryResponse)  # Should still return response
        assert len(response.generated_answer) > 0  # Should have some answer text

    def test_generate_response_truncates_long_answers(
        self, prompt_builder: PromptBuilder
    ) -> None:
        """Test that excessively long answers are truncated."""
        # Create template that produces very long output
        long_template = TemplateEngine({"response": "x" * 1000})  # Long template
        generator = ResponseGenerator(prompt_builder, long_template, response_max_length=100)
        result = QueryResult(  # Simple result
            chunk=Chunk(
                chunk_id="t1", document_id="d1", content="test",
                chunk_index=0, start_char=0, end_char=4, metadata={},
            ),
            similarity_score=0.9, rank=1,
        )
        response = generator.generate_response("test", [result])  # Generate
        assert len(response.generated_answer) <= 100  # Should be within limit

    def test_generate_response_deduplicates_sources(
        self, generator: ResponseGenerator
    ) -> None:
        """Test that duplicate sources are not repeated in the response."""
        results = [  # Multiple results from same source
            QueryResult(
                chunk=Chunk(
                    chunk_id="s1", document_id="d1", content="Chunk 1",
                    chunk_index=0, start_char=0, end_char=7,
                    metadata={"filename": "same_file.txt"},
                ),
                similarity_score=0.9, rank=1,
            ),
            QueryResult(
                chunk=Chunk(
                    chunk_id="s2", document_id="d1", content="Chunk 2",
                    chunk_index=1, start_char=8, end_char=15,
                    metadata={"filename": "same_file.txt"},
                ),
                similarity_score=0.8, rank=2,
            ),
        ]
        response = generator.generate_response("test query", results)  # Generate
        assert response.sources.count("same_file.txt") == 1  # Only one occurrence
