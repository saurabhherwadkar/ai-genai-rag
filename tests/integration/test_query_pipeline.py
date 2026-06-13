# =============================================================================
# RAG Pipeline - Query Pipeline Integration Tests
# =============================================================================
# End-to-end tests for the complete query flow.

from unittest.mock import MagicMock  # Import MagicMock for mocking

import pytest  # Import pytest for test decorators and assertions

from rag_pipeline.generation.prompt_builder import PromptBuilder  # Prompt builder
from rag_pipeline.generation.response_generator import ResponseGenerator  # Generator
from rag_pipeline.generation.template_engine import TemplateEngine  # Template engine
from rag_pipeline.ingestion.embedding_generator import EmbeddingGenerator  # Embedding
from rag_pipeline.models.chunk import Chunk  # Chunk model
from rag_pipeline.models.pipeline_config import EmbeddingConfig, RetrievalConfig  # Configs
from rag_pipeline.query.query_pipeline import QueryPipeline  # Pipeline under test
from rag_pipeline.query.query_processor import QueryProcessor  # Processor
from rag_pipeline.query.reranker import Reranker  # Reranker
from rag_pipeline.query.retriever import Retriever  # Retriever
from rag_pipeline.security.input_sanitizer import InputSanitizer  # Sanitizer
from rag_pipeline.vectorstore.vector_store_manager import VectorStoreManager  # Store


class TestQueryPipelineIntegration:
    """Integration tests for the complete query pipeline."""

    @pytest.fixture  # Fixture for creating the full query pipeline
    def pipeline_with_data(self) -> QueryPipeline:
        """Create a complete query pipeline with pre-loaded test data."""
        import numpy as np  # Import numpy for fake embeddings

        # Set up vector store with test data
        vector_store = VectorStoreManager(  # Create in-memory store
            collection_name="test_query",
            persist_directory=None,
            distance_metric="cosine",
        )
        vector_store.initialize_store()  # Initialize store
        # Add test chunks with embeddings
        test_chunks = [  # Pre-built test chunks
            Chunk(
                chunk_id="q-chunk-1", document_id="qdoc-1",
                content="Machine learning uses algorithms to find patterns in data.",
                chunk_index=0, start_char=0, end_char=58,
                metadata={"filename": "ml_guide.txt", "document_id": "qdoc-1"},
                embedding=[0.1, 0.2, 0.3] * 128,  # 384-dim
            ),
            Chunk(
                chunk_id="q-chunk-2", document_id="qdoc-1",
                content="Deep learning is a subset that uses neural networks with layers.",
                chunk_index=1, start_char=59, end_char=122,
                metadata={"filename": "ml_guide.txt", "document_id": "qdoc-1"},
                embedding=[0.4, 0.5, 0.6] * 128,  # 384-dim
            ),
            Chunk(
                chunk_id="q-chunk-3", document_id="qdoc-2",
                content="Python is a popular programming language for data science.",
                chunk_index=0, start_char=0, end_char=57,
                metadata={"filename": "python_guide.txt", "document_id": "qdoc-2"},
                embedding=[0.7, 0.8, 0.9] * 128,  # 384-dim
            ),
        ]
        vector_store.add_chunks(test_chunks)  # Add chunks to store
        # Create embedding generator with mock model
        embedding_config = EmbeddingConfig(model_name="all-MiniLM-L6-v2")
        embedding_generator = EmbeddingGenerator(embedding_config)
        mock_model = MagicMock()  # Mock model
        mock_model.encode.return_value = np.array([0.1, 0.2, 0.3] * 128)  # Query embedding
        embedding_generator._model = mock_model  # Inject mock
        # Create retrieval config
        retrieval_config = RetrievalConfig(
            top_k=3, similarity_threshold=0.0, rerank_enabled=True, max_query_length=500
        )
        # Assemble pipeline components
        input_sanitizer = InputSanitizer(max_input_length=1000)  # Sanitizer
        query_processor = QueryProcessor(retrieval_config, input_sanitizer)  # Processor
        retriever = Retriever(vector_store, embedding_generator, retrieval_config)  # Retriever
        reranker = Reranker(retrieval_config)  # Reranker
        prompt_builder = PromptBuilder(  # Prompt builder
            system_template="Answer based on context.",
            user_template="Context:\n{context}\n\nQuestion: {query}\n\nAnswer:",
        )
        template_engine = TemplateEngine({  # Template engine
            "response": "Based on the context:\n{context}\n\nFor query: {query}"
        })
        response_generator = ResponseGenerator(  # Generator
            prompt_builder=prompt_builder,
            template_engine=template_engine,
            response_max_length=1000,
        )
        return QueryPipeline(  # Return assembled pipeline
            query_processor=query_processor,
            retriever=retriever,
            reranker=reranker,
            response_generator=response_generator,
        )

    def test_query_returns_response_with_answer(
        self, pipeline_with_data: QueryPipeline
    ) -> None:
        """Test that a query returns a response with a non-empty answer."""
        response = pipeline_with_data.execute_query("What is machine learning?")
        assert response is not None  # Should return a response
        assert len(response.generated_answer) > 0  # Answer should not be empty

    def test_query_preserves_original_query(
        self, pipeline_with_data: QueryPipeline
    ) -> None:
        """Test that the original query is preserved in the response."""
        query = "How does deep learning work?"  # Test query
        response = pipeline_with_data.execute_query(query)  # Execute
        assert response.original_query == query  # Original preserved

    def test_query_includes_sources(
        self, pipeline_with_data: QueryPipeline
    ) -> None:
        """Test that the response includes source attributions."""
        response = pipeline_with_data.execute_query("What is machine learning?")
        assert len(response.sources) > 0  # Should have sources

    def test_query_records_latency(
        self, pipeline_with_data: QueryPipeline
    ) -> None:
        """Test that pipeline latency is recorded in the response."""
        response = pipeline_with_data.execute_query("What is ML?")  # Execute
        assert response.latency_ms > 0  # Latency should be positive

    def test_query_returns_ranked_results(
        self, pipeline_with_data: QueryPipeline
    ) -> None:
        """Test that results are ranked in the response."""
        response = pipeline_with_data.execute_query("machine learning algorithms")
        if response.results:  # If results were returned
            ranks = [r.rank for r in response.results]  # Extract ranks
            assert ranks == sorted(ranks)  # Ranks should be in order
