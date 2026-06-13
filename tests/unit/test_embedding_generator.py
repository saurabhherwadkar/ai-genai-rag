# =============================================================================
# RAG Pipeline - Embedding Generator Unit Tests
# =============================================================================
# Tests for the EmbeddingGenerator class with mocked model.

from unittest.mock import MagicMock  # Import mocking utilities

import pytest  # Import pytest for test decorators and assertions

from rag_pipeline.ingestion.embedding_generator import EmbeddingGenerator  # Class under test
from rag_pipeline.models.chunk import Chunk  # Import Chunk for test data
from rag_pipeline.models.pipeline_config import EmbeddingConfig  # Import config model
from rag_pipeline.utils.exceptions import EmbeddingError  # Expected exception


class TestEmbeddingGenerator:
    """Unit tests for the EmbeddingGenerator class."""

    @pytest.fixture  # Fixture for creating a mock embedding model
    def mock_model(self):
        """Create a mock SentenceTransformer model."""
        import numpy as np  # Import numpy for creating fake embeddings

        model = MagicMock()  # Create mock model object
        # Configure encode to return fake embeddings
        model.encode.return_value = np.array([[0.1] * 384])  # Fake 384-dim embedding
        model.get_sentence_embedding_dimension.return_value = 384  # Fake dimension
        return model  # Return the mock model

    @pytest.fixture  # Fixture for generator with mocked model
    def generator(self, embedding_config: EmbeddingConfig, mock_model) -> EmbeddingGenerator:
        """Create an EmbeddingGenerator with a mocked model."""
        gen = EmbeddingGenerator(embedding_config)  # Create generator
        gen._model = mock_model  # Inject mock model (skip actual loading)
        return gen  # Return the configured generator

    def test_generate_embedding_returns_list_of_floats(
        self, generator: EmbeddingGenerator, mock_model
    ) -> None:
        """Test that generate_embedding returns a list of float values."""
        import numpy as np  # Import numpy for mock return value

        mock_model.encode.return_value = np.array([0.1] * 384)  # Single embedding
        result = generator.generate_embedding("Test text")  # Generate embedding
        assert isinstance(result, list)  # Should return a list
        assert all(isinstance(v, float) for v in result)  # All values should be floats

    def test_generate_embedding_returns_correct_dimension(
        self, generator: EmbeddingGenerator, mock_model
    ) -> None:
        """Test that embedding dimension matches expected model output."""
        import numpy as np  # Import numpy for mock return value

        mock_model.encode.return_value = np.array([0.5] * 384)  # 384-dim embedding
        result = generator.generate_embedding("Test text")  # Generate embedding
        assert len(result) == 384  # Should be 384 dimensions

    def test_generate_embedding_raises_for_empty_text(
        self, generator: EmbeddingGenerator
    ) -> None:
        """Test that empty text input raises EmbeddingError."""
        with pytest.raises(EmbeddingError):  # Expect EmbeddingError
            generator.generate_embedding("")  # Pass empty string

    def test_generate_embedding_raises_for_whitespace_text(
        self, generator: EmbeddingGenerator
    ) -> None:
        """Test that whitespace-only text raises EmbeddingError."""
        with pytest.raises(EmbeddingError):  # Expect EmbeddingError
            generator.generate_embedding("   \t\n  ")  # Pass whitespace only

    def test_generate_embeddings_batch_returns_correct_count(
        self, generator: EmbeddingGenerator, mock_model
    ) -> None:
        """Test that batch embedding returns one vector per input text."""
        import numpy as np  # Import numpy for mock return value

        texts = ["Text one", "Text two", "Text three"]  # Three test texts
        mock_model.encode.return_value = np.array([[0.1] * 384] * 3)  # Three embeddings
        results = generator.generate_embeddings_batch(texts)  # Generate batch
        assert len(results) == 3  # Should return 3 embeddings

    def test_generate_embeddings_batch_empty_list(
        self, generator: EmbeddingGenerator
    ) -> None:
        """Test that an empty input list returns an empty result list."""
        results = generator.generate_embeddings_batch([])  # Pass empty list
        assert results == []  # Should return empty list

    def test_embed_chunks_returns_chunks_with_embeddings(
        self, generator: EmbeddingGenerator, sample_chunks: list[Chunk], mock_model
    ) -> None:
        """Test that embed_chunks attaches embeddings to chunk objects."""
        import numpy as np  # Import numpy for mock return value

        mock_model.encode.return_value = np.array(  # Mock batch embeddings
            [[0.1] * 384] * len(sample_chunks)  # One embedding per chunk
        )
        embedded = generator.embed_chunks(sample_chunks)  # Embed all chunks
        assert len(embedded) == len(sample_chunks)  # Same count as input
        for chunk in embedded:  # Check each embedded chunk
            assert chunk.has_embedding  # Should have embedding attached
            assert len(chunk.embedding) == 384  # Should be 384 dimensions

    def test_embed_chunks_preserves_chunk_content(
        self, generator: EmbeddingGenerator, sample_chunks: list[Chunk], mock_model
    ) -> None:
        """Test that embedding does not alter the chunk content."""
        import numpy as np  # Import numpy for mock return value

        mock_model.encode.return_value = np.array(  # Mock batch embeddings
            [[0.1] * 384] * len(sample_chunks)  # One per chunk
        )
        embedded = generator.embed_chunks(sample_chunks)  # Embed chunks
        for original, embedded_chunk in zip(sample_chunks, embedded, strict=True):  # Compare
            assert embedded_chunk.content == original.content  # Content unchanged
            assert embedded_chunk.chunk_id == original.chunk_id  # ID unchanged

    def test_get_embedding_dimension(
        self, generator: EmbeddingGenerator, mock_model
    ) -> None:
        """Test that get_embedding_dimension returns the model's dimension."""
        dimension = generator.get_embedding_dimension()  # Query dimension
        assert dimension == 384  # Should match mock model's dimension
