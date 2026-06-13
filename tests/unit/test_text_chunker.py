# =============================================================================
# RAG Pipeline - Text Chunker Unit Tests
# =============================================================================
# Tests for the TextChunker class covering all chunking scenarios.

import pytest  # Import pytest for test decorators and assertions

from rag_pipeline.ingestion.text_chunker import TextChunker  # Class under test
from rag_pipeline.models.chunk import Chunk  # Import Chunk for type checking
from rag_pipeline.models.document import Document  # Import Document for test data
from rag_pipeline.models.pipeline_config import ChunkingConfig  # Import config model
from rag_pipeline.utils.exceptions import ChunkingError  # Expected exception


class TestTextChunker:
    """Unit tests for the TextChunker class."""

    @pytest.fixture  # Fixture for creating a TextChunker instance
    def chunker(self, chunking_config: ChunkingConfig) -> TextChunker:
        """Create a TextChunker with test configuration."""
        return TextChunker(chunking_config)  # Use the shared chunking_config fixture

    def test_chunk_document_returns_list_of_chunks(
        self, chunker: TextChunker, sample_document: Document
    ) -> None:
        """Test that chunking a document returns a non-empty list of Chunks."""
        chunks = chunker.chunk_document(sample_document)  # Chunk the document
        assert isinstance(chunks, list)  # Should return a list
        assert len(chunks) > 0  # Should have at least one chunk
        assert all(isinstance(c, Chunk) for c in chunks)  # All items should be Chunks

    def test_chunk_document_preserves_document_id(
        self, chunker: TextChunker, sample_document: Document
    ) -> None:
        """Test that all chunks reference the parent document ID."""
        chunks = chunker.chunk_document(sample_document)  # Chunk the document
        for chunk in chunks:  # Check each chunk
            assert chunk.document_id == sample_document.document_id  # Should match parent

    def test_chunk_document_assigns_sequential_indices(
        self, chunker: TextChunker, sample_document: Document
    ) -> None:
        """Test that chunks receive sequential zero-based indices."""
        chunks = chunker.chunk_document(sample_document)  # Chunk the document
        indices = [chunk.chunk_index for chunk in chunks]  # Extract all indices
        assert indices == list(range(len(chunks)))  # Should be 0, 1, 2, ...

    def test_chunk_document_assigns_unique_ids(
        self, chunker: TextChunker, sample_document: Document
    ) -> None:
        """Test that each chunk receives a unique identifier."""
        chunks = chunker.chunk_document(sample_document)  # Chunk the document
        chunk_ids = [chunk.chunk_id for chunk in chunks]  # Extract all IDs
        assert len(chunk_ids) == len(set(chunk_ids))  # All IDs should be unique

    def test_chunk_document_raises_for_empty_content(self, chunker: TextChunker) -> None:
        """Test that chunking an empty document raises ChunkingError."""
        empty_doc = Document(  # Create document with empty content
            content="",  # Empty content
            source_path="/test/empty.txt",  # Mock path
            document_id="empty-doc",  # Mock ID
        )
        with pytest.raises(ChunkingError):  # Expect ChunkingError
            chunker.chunk_document(empty_doc)  # Attempt to chunk

    def test_chunk_document_raises_for_whitespace_content(
        self, chunker: TextChunker
    ) -> None:
        """Test that chunking whitespace-only document raises ChunkingError."""
        whitespace_doc = Document(  # Create document with only whitespace
            content="   \n\n  \t  ",  # Whitespace only
            source_path="/test/whitespace.txt",  # Mock path
            document_id="ws-doc",  # Mock ID
        )
        with pytest.raises(ChunkingError):  # Expect ChunkingError
            chunker.chunk_document(whitespace_doc)  # Attempt to chunk

    def test_chunk_splits_on_separator(self) -> None:
        """Test that text is split on the configured separator."""
        config = ChunkingConfig(chunk_size=500, chunk_overlap=0, separator="\n\n", min_chunk_size=5)
        chunker = TextChunker(config)  # Create chunker with no overlap
        doc = Document(  # Document with clear paragraph separators
            content="First paragraph.\n\nSecond paragraph.\n\nThird paragraph.",
            source_path="/test.txt",
            document_id="test-split",
        )
        chunks = chunker.chunk_document(doc)  # Chunk the document
        assert len(chunks) == 3  # Should produce 3 chunks (one per paragraph)

    def test_small_chunks_are_merged(self) -> None:
        """Test that chunks below minimum size are merged with neighbors."""
        config = ChunkingConfig(
            chunk_size=500, chunk_overlap=0, separator="\n\n", min_chunk_size=30
        )
        chunker = TextChunker(config)  # Create chunker with higher minimum
        doc = Document(  # Document with one very short paragraph
            content="OK.\n\nThis is a sufficiently long second paragraph for testing.",
            source_path="/test.txt",
            document_id="test-merge",
        )
        chunks = chunker.chunk_document(doc)  # Chunk the document
        # The short "OK." should be merged with the next paragraph
        assert len(chunks) == 1  # Should produce 1 merged chunk

    def test_single_paragraph_returns_one_chunk(self) -> None:
        """Test that a document with no separators returns a single chunk."""
        config = ChunkingConfig(
            chunk_size=1000, chunk_overlap=0, separator="\n\n", min_chunk_size=5
        )
        chunker = TextChunker(config)  # Create chunker with large chunk size
        doc = Document(  # Single paragraph document
            content="This is one continuous paragraph with no line breaks at all.",
            source_path="/test.txt",
            document_id="test-single",
        )
        chunks = chunker.chunk_document(doc)  # Chunk the document
        assert len(chunks) == 1  # Should produce exactly one chunk

    def test_chunk_content_not_empty(
        self, chunker: TextChunker, sample_document: Document
    ) -> None:
        """Test that no produced chunks have empty content."""
        chunks = chunker.chunk_document(sample_document)  # Chunk the document
        for chunk in chunks:  # Check each chunk
            assert len(chunk.content.strip()) > 0  # Content should not be empty
