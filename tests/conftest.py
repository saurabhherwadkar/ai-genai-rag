# =============================================================================
# RAG Pipeline - Shared Test Fixtures
# =============================================================================
# Provides reusable fixtures for all test modules in the test suite.

from pathlib import Path  # Import Path for file system operations

import pytest  # Import pytest for fixture decorators

from rag_pipeline.models.chunk import Chunk  # Import Chunk data model
from rag_pipeline.models.document import Document  # Import Document data model
from rag_pipeline.models.pipeline_config import (  # Import configuration models
    ChunkingConfig,
    EmbeddingConfig,
    RetrievalConfig,
)


@pytest.fixture  # Register as a pytest fixture for dependency injection
def sample_document() -> Document:
    """Create a sample Document object for testing.

    Returns:
        Document with realistic content for testing chunking and embedding.
    """
    content = (  # Multi-paragraph text that can be split into chunks
        "Machine learning is a subset of artificial intelligence that focuses on "
        "building systems that learn from data. Instead of being explicitly programmed, "
        "these systems identify patterns and make decisions with minimal human intervention.\n\n"
        "Deep learning is a further subset of machine learning that uses neural networks "
        "with many layers. These deep neural networks can learn complex patterns in large "
        "amounts of data, enabling breakthroughs in image recognition and natural language "
        "processing.\n\n"
        "Reinforcement learning is another approach where an agent learns to make decisions "
        "by performing actions in an environment and receiving rewards or penalties. This "
        "technique has been successfully applied to game playing and robotics."
    )
    return Document(  # Create and return the Document instance
        content=content,  # Set the test content
        source_path="/test/sample_document.txt",  # Mock source path
        document_id="test-doc-001",  # Known test ID for assertions
        metadata={"filename": "sample_document.txt", "extension": ".txt"},  # Test metadata
        content_hash="abc123hash",  # Mock content hash
    )


@pytest.fixture  # Register as a pytest fixture
def sample_chunks(sample_document: Document) -> list[Chunk]:
    """Create a list of sample Chunk objects for testing.

    Args:
        sample_document: The sample document fixture to reference.

    Returns:
        List of Chunk objects simulating chunked document content.
    """
    return [  # Return a list of test chunks
        Chunk(  # First chunk about machine learning
            chunk_id="chunk-001",  # Known chunk ID for assertions
            document_id=sample_document.document_id,  # Reference parent document
            content="Machine learning is a subset of artificial intelligence.",  # Chunk text
            chunk_index=0,  # First chunk position
            start_char=0,  # Starts at beginning
            end_char=55,  # End character position
            metadata={"filename": "sample_document.txt"},  # Inherited metadata
        ),
        Chunk(  # Second chunk about deep learning
            chunk_id="chunk-002",  # Known chunk ID
            document_id=sample_document.document_id,  # Reference parent document
            content="Deep learning uses neural networks with many layers.",  # Chunk text
            chunk_index=1,  # Second chunk position
            start_char=56,  # Continues from first chunk
            end_char=107,  # End character position
            metadata={"filename": "sample_document.txt"},  # Inherited metadata
        ),
        Chunk(  # Third chunk about reinforcement learning
            chunk_id="chunk-003",  # Known chunk ID
            document_id=sample_document.document_id,  # Reference parent document
            content="Reinforcement learning uses rewards and penalties.",  # Chunk text
            chunk_index=2,  # Third chunk position
            start_char=108,  # Continues from second chunk
            end_char=157,  # End character position
            metadata={"filename": "sample_document.txt"},  # Inherited metadata
        ),
    ]


@pytest.fixture  # Register as a pytest fixture
def sample_embedding() -> list[float]:
    """Create a sample embedding vector for testing.

    Returns:
        List of floats simulating a 384-dimension embedding (shortened for tests).
    """
    # Return a small fake embedding (real model produces 384 dimensions)
    return [0.1] * 384  # Simple uniform vector for testing


@pytest.fixture  # Register as a pytest fixture
def chunking_config() -> ChunkingConfig:
    """Create a ChunkingConfig with test-friendly settings.

    Returns:
        ChunkingConfig with smaller sizes suitable for test content.
    """
    return ChunkingConfig(  # Create test-appropriate chunking config
        chunk_size=100,  # Small chunk size for test documents
        chunk_overlap=20,  # Small overlap for testing overlap logic
        separator="\n\n",  # Standard paragraph separator
        min_chunk_size=10,  # Low minimum for testing edge cases
    )


@pytest.fixture  # Register as a pytest fixture
def embedding_config() -> EmbeddingConfig:
    """Create an EmbeddingConfig for testing.

    Returns:
        EmbeddingConfig with the default model settings.
    """
    return EmbeddingConfig(  # Create embedding config
        model_name="all-MiniLM-L6-v2",  # Standard test model
        batch_size=8,  # Small batch for tests
        normalize=True,  # Enable normalization
    )


@pytest.fixture  # Register as a pytest fixture
def retrieval_config() -> RetrievalConfig:
    """Create a RetrievalConfig for testing.

    Returns:
        RetrievalConfig with test-appropriate settings.
    """
    return RetrievalConfig(  # Create retrieval config
        top_k=3,  # Small top_k for tests
        similarity_threshold=0.2,  # Low threshold to allow test results
        rerank_enabled=True,  # Enable reranking for testing
        max_query_length=500,  # Reasonable limit for tests
    )


@pytest.fixture  # Register as a pytest fixture
def mock_settings() -> dict:
    """Create a mock settings dictionary for testing components.

    Returns:
        Dictionary mimicking the structure of loaded settings.yaml.
    """
    return {  # Build mock settings structure
        "application": {  # Application section
            "name": "Test RAG Pipeline",  # Test app name
            "version": "1.0.0",  # Test version
            "environment": "test",  # Test environment
        },
        "ingestion": {  # Ingestion section
            "chunking": {  # Chunking settings
                "chunk_size": 100,  # Test chunk size
                "chunk_overlap": 20,  # Test overlap
                "separator": "\n\n",  # Paragraph separator
                "min_chunk_size": 10,  # Test minimum
            },
            "embedding": {  # Embedding settings
                "model_name": "all-MiniLM-L6-v2",  # Model name
                "batch_size": 8,  # Small batch
                "normalize": True,  # Normalize embeddings
            },
            "supported_extensions": [".txt", ".md"],  # Test extensions
        },
        "vectorstore": {  # Vector store section
            "collection_name": "test_collection",  # Test collection
            "persist_directory": None,  # In-memory for tests
            "distance_metric": "cosine",  # Distance metric
        },
        "retrieval": {  # Retrieval section
            "top_k": 3,  # Test top_k
            "similarity_threshold": 0.2,  # Low threshold
            "rerank_enabled": True,  # Enable reranking
            "max_query_length": 500,  # Max query length
        },
        "security": {  # Security section
            "max_input_length": 1000,  # Test max length
        },
    }


@pytest.fixture  # Register as a pytest fixture
def tmp_text_file(tmp_path: Path) -> Path:
    """Create a temporary text file for document loading tests.

    Args:
        tmp_path: Pytest's built-in temporary directory fixture.

    Returns:
        Path to the created temporary text file.
    """
    content = "This is a test document.\n\nIt has multiple paragraphs.\n\nThird paragraph here."
    file_path = tmp_path / "test_document.txt"  # Define file path
    file_path.write_text(content, encoding="utf-8")  # Write test content
    return file_path  # Return the file path


@pytest.fixture  # Register as a pytest fixture
def tmp_config_dir(tmp_path: Path) -> Path:
    """Create a temporary config directory with test settings.

    Args:
        tmp_path: Pytest's built-in temporary directory fixture.

    Returns:
        Path to the created temporary config directory.
    """
    config_dir = tmp_path / "config"  # Define config directory path
    config_dir.mkdir()  # Create the directory
    settings_content = (  # Minimal settings YAML for testing
        "application:\n"
        "  name: Test App\n"
        "  environment: test\n"
        "ingestion:\n"
        "  chunking:\n"
        "    chunk_size: 100\n"
    )
    settings_file = config_dir / "settings.yaml"  # Define settings file path
    settings_file.write_text(settings_content, encoding="utf-8")  # Write settings
    return config_dir  # Return the config directory path
