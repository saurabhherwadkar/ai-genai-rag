# =============================================================================
# RAG Pipeline - Vector Store Manager Unit Tests
# =============================================================================
# Tests for the VectorStoreManager class with in-memory ChromaDB.

import pytest  # Import pytest for test decorators and assertions

from rag_pipeline.models.chunk import Chunk  # Import Chunk for test data
from rag_pipeline.utils.exceptions import VectorStoreError  # Expected exception
from rag_pipeline.vectorstore.vector_store_manager import VectorStoreManager  # Class under test


class TestVectorStoreManager:
    """Unit tests for the VectorStoreManager class."""

    @pytest.fixture  # Fixture for creating an initialized vector store
    def store(self) -> VectorStoreManager:
        """Create and initialize an in-memory VectorStoreManager."""
        manager = VectorStoreManager(  # Create manager with test settings
            collection_name="test_collection",  # Test collection name
            persist_directory=None,  # In-memory mode
            distance_metric="cosine",  # Cosine similarity
        )
        manager.initialize_store()  # Initialize the store
        return manager  # Return the initialized store

    @pytest.fixture  # Fixture for chunks with embeddings
    def chunks_with_embeddings(self) -> list[Chunk]:
        """Create test chunks with fake embedding vectors."""
        return [  # Return list of chunks with embeddings
            Chunk(  # First test chunk
                chunk_id="test-chunk-1",
                document_id="doc-001",
                content="Machine learning basics",
                chunk_index=0,
                start_char=0,
                end_char=23,
                metadata={"filename": "test.txt"},
                embedding=[0.1, 0.2, 0.3] * 128,  # 384-dim fake embedding
            ),
            Chunk(  # Second test chunk
                chunk_id="test-chunk-2",
                document_id="doc-001",
                content="Deep learning with neural networks",
                chunk_index=1,
                start_char=24,
                end_char=58,
                metadata={"filename": "test.txt"},
                embedding=[0.4, 0.5, 0.6] * 128,  # 384-dim fake embedding
            ),
        ]

    def test_initialize_store_creates_collection(self, store: VectorStoreManager) -> None:
        """Test that initialization creates the vector store collection."""
        stats = store.get_collection_stats()  # Get collection stats
        assert stats["collection_name"] == "test_collection"  # Name should match
        assert stats["total_chunks"] == 0  # Should start empty

    def test_add_chunks_stores_in_collection(
        self, store: VectorStoreManager, chunks_with_embeddings: list[Chunk]
    ) -> None:
        """Test that adding chunks increases the collection count."""
        count = store.add_chunks(chunks_with_embeddings)  # Add chunks
        assert count == 2  # Should have added 2 chunks
        stats = store.get_collection_stats()  # Check stats
        assert stats["total_chunks"] == 2  # Collection should have 2 items

    def test_add_chunks_skips_without_embeddings(self, store: VectorStoreManager) -> None:
        """Test that chunks without embeddings are not stored."""
        chunks_no_embedding = [  # Chunks without embeddings
            Chunk(
                chunk_id="no-embed-1",
                document_id="doc-002",
                content="No embedding here",
                chunk_index=0,
                start_char=0,
                end_char=18,
                metadata={},
            ),
        ]
        count = store.add_chunks(chunks_no_embedding)  # Attempt to add
        assert count == 0  # Should not store any chunks

    def test_add_chunks_empty_list(self, store: VectorStoreManager) -> None:
        """Test that adding an empty list returns zero."""
        count = store.add_chunks([])  # Add empty list
        assert count == 0  # Should return zero

    def test_search_similar_returns_results(
        self, store: VectorStoreManager, chunks_with_embeddings: list[Chunk]
    ) -> None:
        """Test that similarity search returns relevant results."""
        store.add_chunks(chunks_with_embeddings)  # Add test chunks first
        query_embedding = [0.1, 0.2, 0.3] * 128  # Similar to first chunk
        results = store.search_similar(query_embedding, top_k=2)  # Search
        assert len(results) > 0  # Should return at least one result
        assert results[0].chunk.content is not None  # Should have content

    def test_search_similar_respects_top_k(
        self, store: VectorStoreManager, chunks_with_embeddings: list[Chunk]
    ) -> None:
        """Test that search returns at most top_k results."""
        store.add_chunks(chunks_with_embeddings)  # Add 2 chunks
        query_embedding = [0.1, 0.2, 0.3] * 128  # Query embedding
        results = store.search_similar(query_embedding, top_k=1)  # Request only 1
        assert len(results) <= 1  # Should return at most 1

    def test_delete_document_removes_chunks(
        self, store: VectorStoreManager, chunks_with_embeddings: list[Chunk]
    ) -> None:
        """Test that deleting a document removes all its chunks."""
        store.add_chunks(chunks_with_embeddings)  # Add chunks
        deleted = store.delete_document("doc-001")  # Delete by document ID
        assert deleted == 2  # Should delete both chunks
        stats = store.get_collection_stats()  # Check stats
        assert stats["total_chunks"] == 0  # Collection should be empty

    def test_clear_collection_empties_store(
        self, store: VectorStoreManager, chunks_with_embeddings: list[Chunk]
    ) -> None:
        """Test that clear_collection removes all items."""
        store.add_chunks(chunks_with_embeddings)  # Add chunks
        store.clear_collection()  # Clear the collection
        stats = store.get_collection_stats()  # Check stats
        assert stats["total_chunks"] == 0  # Should be empty

    def test_operations_before_init_raise_error(self) -> None:
        """Test that operations on uninitialized store raise VectorStoreError."""
        store = VectorStoreManager("test", None, "cosine")  # Create but don't initialize
        with pytest.raises(VectorStoreError):  # Expect error
            store.add_chunks([])  # Attempt operation without initialization

    def test_get_collection_stats_returns_expected_keys(
        self, store: VectorStoreManager
    ) -> None:
        """Test that stats dictionary contains all expected keys."""
        stats = store.get_collection_stats()  # Get stats
        assert "collection_name" in stats  # Should have name
        assert "total_chunks" in stats  # Should have count
        assert "distance_metric" in stats  # Should have metric
        assert "persist_directory" in stats  # Should have persist info
